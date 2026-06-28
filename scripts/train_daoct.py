#!/usr/bin/env python3
"""
Treino portável do DA-OCT Challenge.

Roda igual em:
  - Mac mini (CPU/MPS)          -> sanity rápido, dev
  - Kaggle T4 (CUDA, fp16)      -> treino médio
  - RTX 5090 (Blackwell, bf16)  -> treino pesado

Seleciona device e precisão (AMP) automaticamente; nada é hardcoded por GPU.
Salva o checkpoint com o nome que o `main.py`/`infer_test_monai.py` do kit reconhecem
(`unet_maestro2_semi.pth`), mantendo a submissão drop-in.

Exemplos:
  # sanity no Mac (poucas amostras, poucas épocas)
  python scripts/train_daoct.py --data_root data/starter_kit/app_ingestion/input_data/train \
      --out models --epochs 2 --limit 24 --workers 0

  # pesado na 5090
  python scripts/train_daoct.py --data_root <DATA>/train --out models \
      --epochs 150 --batch_size 16 --amp auto --aug strong --semi

A arquitetura padrão é IGUAL à do baseline (UNet 16/32/64/128) para o checkpoint ser
drop-in no infer do kit. Se mudar --channels, é preciso atualizar o build_model do infer.
"""
import argparse, glob, json, os, random, sys, time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from monai.data import CacheDataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceCELoss, HausdorffDTLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord,
    RandFlipd, RandRotate90d, RandAffined, RandGridDistortiond, Rand2DElasticd,
    RandScaleIntensityd, RandShiftIntensityd, RandAdjustContrastd,
    RandGaussianNoised, RandGaussianSmoothd,
)

DEFAULT_CHANNELS = (16, 32, 64, 128)
DEFAULT_STRIDES = (2, 2, 2)
SUPERVISED_DEVICE = "Topcon_Maestro2"
UNLABELED_DEVICES = ["Heidelberg_Spectralis", "Zeiss_Cirrus", "Topcon_Maestro2_unlabeled"]
NUM_CLASSES = 10


# --------------------------- portabilidade -----------------------------------
def pick_device(opt: str) -> torch.device:
    if opt != "auto":
        return torch.device(opt)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_amp(amp_opt: str, device: torch.device):
    """Retorna (enabled, dtype, use_scaler). AMP só em CUDA."""
    if amp_opt == "off" or device.type != "cuda":
        return False, None, False
    bf16_ok = torch.cuda.is_bf16_supported()
    if amp_opt == "bf16" or (amp_opt == "auto" and bf16_ok):
        return True, torch.bfloat16, False          # bf16: sem GradScaler
    return True, torch.float16, True                 # fp16 (T4): com GradScaler


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------- dados -------------------------------------------
def list_supervised(root: Path):
    items = []
    for status in ["Diseased", "Healthy"]:
        for img in sorted(glob.glob(str(root / SUPERVISED_DEVICE / status / "*-image.png"))):
            mask = img.replace("-image.png", "-mask.png")
            if os.path.exists(mask):
                items.append({"image": img, "label": mask})
    return items


def build_transforms(img_size: int, aug: str, train: bool):
    keys = ["image", "label"]
    base = [
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityd(keys="image"),
        Resized(keys="image", spatial_size=(img_size, img_size), mode="bilinear"),
        Resized(keys="label", spatial_size=(img_size, img_size), mode="nearest"),
    ]
    if train and aug != "none":
        spatial = [
            RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
            RandRotate90d(keys=keys, prob=0.3, max_k=1),
        ]
        # augmentação de APARÊNCIA (só na imagem) = robustez cross-vendor
        appearance = [
            RandScaleIntensityd(keys="image", factors=0.3, prob=0.6),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            RandAdjustContrastd(keys="image", gamma=(0.6, 1.7), prob=0.6),
            RandGaussianNoised(keys="image", prob=0.3, std=0.04),
            RandGaussianSmoothd(keys="image", prob=0.3),
        ]
        if aug in ("strong", "widefield", "widefield2"):
            aff = dict(prob=0.4, rotate_range=0.15, shear_range=0.1, scale_range=0.15)
            if aug in ("widefield", "widefield2"):
                # generalização geométrica p/ wide-field (curvatura/FOV forte): grid distortion +
                # affine grande (+ elástica no widefield2). Valida no proxy (eval_proxy_widefield.py).
                aff = dict(prob=0.6, rotate_range=0.30, shear_range=0.2, scale_range=0.35)
                gd = dict(num_cells=6, distort_limit=0.3) if aug == "widefield" else dict(num_cells=8, distort_limit=0.4)
                spatial.append(RandGridDistortiond(
                    keys=keys, prob=0.6, mode=("bilinear", "nearest"), padding_mode="border", **gd,
                ))
                if aug == "widefield2":
                    spatial.append(Rand2DElasticd(
                        keys=keys, prob=0.5, spacing=(30, 30), magnitude_range=(2, 5),
                        mode=("bilinear", "nearest"), padding_mode="border",
                    ))
            spatial.append(RandAffined(
                keys=keys, mode=("bilinear", "nearest"), padding_mode="border", **aff,
            ))
        base += spatial + appearance
    base.append(ToTensord(keys=keys))
    return Compose(base)


# --------------------------- modelo ------------------------------------------
def build_model(channels, device):
    strides = tuple(2 for _ in range(len(channels) - 1))
    return UNet(
        spatial_dims=2, in_channels=1, out_channels=NUM_CLASSES,
        channels=tuple(channels), strides=strides, num_res_units=2,
    ).to(device)


# --------------------------- val ---------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, amp_enabled, amp_dtype):
    model.eval()
    metric = DiceMetric(include_background=True, reduction="mean")
    metric.reset()
    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else nullcontext()
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device).long()
        with ctx:
            logits = model(x)
        pred = torch.argmax(logits, dim=1)
        pred_oh = torch.nn.functional.one_hot(pred, NUM_CLASSES).permute(0, 3, 1, 2).float()
        y_oh = torch.nn.functional.one_hot(y.squeeze(1), NUM_CLASSES).permute(0, 3, 1, 2).float()
        metric(y_pred=pred_oh, y=y_oh)
    return float(metric.aggregate().item())


# --------------------------- main --------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="pasta .../input_data/train")
    ap.add_argument("--out", default="models", help="onde salvar checkpoint/report")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", default=",".join(map(str, DEFAULT_CHANNELS)),
                    help="canais do UNet (default = baseline, drop-in no infer)")
    ap.add_argument("--aug", choices=["none", "basic", "strong", "widefield", "widefield2"], default="strong")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cache_rate", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0, help="cap de amostras p/ sanity (0=tudo)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compile", action="store_true", help="torch.compile (CUDA novas)")
    ap.add_argument("--time_budget_min", type=float, default=0,
                    help="para o treino após N minutos (0=desligado). Útil p/ caber nas 2h do servidor")
    ap.add_argument("--boundary_weight", type=float, default=0.0,
                    help="peso do HausdorffDTLoss (0=off). HD~53x o Dice, então use ~0.005-0.02 (boundary ~0.3-1x do Dice)")
    ap.add_argument("--boundary_warmup_frac", type=float, default=0.2,
                    help="fração das épocas antes de ligar o boundary loss (deixa Dice estabilizar)")
    ap.add_argument("--semi", action="store_true", help="(experimental) pseudo-labels nos não rotulados")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    amp_enabled, amp_dtype, use_scaler = pick_amp(args.amp, device)
    channels = [int(c) for c in args.channels.split(",")]

    print("=" * 60)
    print(f" device      : {device.type}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f" torch       : {torch.__version__}")
    print(f" AMP         : enabled={amp_enabled} dtype={amp_dtype} scaler={use_scaler}")
    print(f" arch        : UNet channels={tuple(channels)}"
          + ("" if channels == list(DEFAULT_CHANNELS) else "  ⚠️ != baseline: atualize o infer!"))
    print(f" img_size    : {args.img_size}  batch={args.batch_size}  epochs={args.epochs}  aug={args.aug}")
    print("=" * 60)

    # dados
    data = list_supervised(Path(args.data_root))
    if not data:
        sys.exit(f"[ERRO] nenhum par image/mask em {args.data_root}/{SUPERVISED_DEVICE}")
    random.shuffle(data)
    if args.limit:
        data = data[: args.limit]
    n_val = max(1, int(len(data) * args.val_frac))
    val_items, train_items = data[:n_val], data[n_val:]
    print(f" supervisionado: {len(train_items)} treino / {len(val_items)} val")

    train_ds = CacheDataset(train_items, build_transforms(args.img_size, args.aug, True),
                            cache_rate=args.cache_rate, num_workers=args.workers)
    val_ds = CacheDataset(val_items, build_transforms(args.img_size, "none", False),
                          cache_rate=args.cache_rate, num_workers=args.workers)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    # modelo / loss / otimizador
    model = build_model(channels, device)
    if args.compile and device.type == "cuda":
        model = torch.compile(model)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    # boundary loss (Hausdorff DT): otimiza a POSIÇÃO da fronteira das camadas = exatamente o MASD.
    # Com warmup: só entra depois que o Dice/CE estabiliza (DT em máscaras ruins é instável).
    loss_bd = HausdorffDTLoss(softmax=True, to_onehot_y=True, include_background=True) if args.boundary_weight > 0 else None
    bd_warmup = int(args.epochs * args.boundary_warmup_frac)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "unet_maestro2_semi.pth")
    # sidecar com a arquitetura -> infer_daoct.py reconstrói o modelo certo p/ qualquer arch
    with open(ckpt_path + ".arch.json", "w") as f:
        json.dump({"channels": channels, "img_size": args.img_size, "num_classes": NUM_CLASSES}, f)
    best_val, history = -1.0, []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device).long()
            opt.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else nullcontext()
            with ctx:
                logits = model(x)
                loss = loss_fn(logits, y)
                if loss_bd is not None and epoch >= bd_warmup:
                    # HausdorffDTLoss usa distance-transform (float64) -> quebra no MPS; cai no CPU lá.
                    # No servidor (CUDA) roda nativo. O autograd lida com a transferência de device.
                    if device.type == "mps":
                        loss = loss + args.boundary_weight * loss_bd(logits.float().cpu(), y.cpu()).to(device)
                    else:
                        loss = loss + args.boundary_weight * loss_bd(logits.float(), y)
            if use_scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            ep_loss += loss.item()
        sched.step()
        val_dice = evaluate(model, val_loader, device, amp_enabled, amp_dtype)
        history.append({"epoch": epoch, "loss": ep_loss / len(train_loader), "val_dice": val_dice})
        flag = ""
        if val_dice > best_val:
            best_val = val_dice
            torch.save(model.state_dict() if not args.compile else model._orig_mod.state_dict(), ckpt_path)
            flag = "  <- best (salvo)"
        elapsed_min = (time.time() - t0) / 60.0
        print(f"[{epoch:3d}/{args.epochs}] loss={ep_loss/len(train_loader):.4f} "
              f"val_dice={val_dice:.4f}{flag}  ({elapsed_min:.1f} min)")
        if args.time_budget_min and elapsed_min >= args.time_budget_min:
            print(f"[INFO] teto de tempo ({args.time_budget_min} min) atingido — parando na época {epoch}.")
            break

    report = {"best_val_dice": best_val, "epochs": args.epochs, "channels": channels,
              "img_size": args.img_size, "aug": args.aug, "amp": str(amp_dtype),
              "device": device.type, "history": history}
    with open(os.path.join(args.out, "train_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] best val_dice={best_val:.4f}  checkpoint={ckpt_path}")
    if args.semi:
        print("[NOTA] --semi ainda é stub: F2 (pseudo-labels cross-vendor) entra na próxima rodada.")


if __name__ == "__main__":
    main()
