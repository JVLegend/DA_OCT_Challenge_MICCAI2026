#!/usr/bin/env python3
"""
F2 — Semi-supervisão por pseudo-labels cross-vendor.

Fluxo em 2 fases:
  1. PSEUDO: carrega o checkpoint teacher (ex: round2), roda softmax em todas as imagens
     não rotuladas (Spectralis, Cirrus, Maestro2_unlabeled), filtra por confiança
     (média do max-prob por pixel >= --conf_threshold) e salva máscaras temporárias.
  2. TRAIN:  re-treina o modelo (warm-start no teacher) misturando Maestro2 rotulado +
     pseudo-rotulados de alta confiança. A cada época o modelo aprende a generalizar
     para os domínios alvo.

Uso:
  python scripts/train_daoct_semi.py \\
    --data_root data/starter_kit/app_ingestion/input_data/train \\
    --teacher    models/round2/unet_maestro2_semi.pth \\
    --out        models/round3_semi \\
    --epochs 150 --batch_size 16 --img_size 384 --channels 32,64,128,256 \\
    --conf_threshold 0.85 --semi_weight 0.5 --amp auto --aug strong
"""
import argparse, glob, json, os, random, sys, tempfile, time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import cv2

from monai.data import CacheDataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized, ToTensord,
    RandFlipd, RandRotate90d, RandAffined, RandGridDistortiond, Rand2DElasticd,
    RandScaleIntensityd, RandShiftIntensityd, RandAdjustContrastd,
    RandGaussianNoised, RandGaussianSmoothd,
    LoadImage, EnsureChannelFirst, ScaleIntensity, Resize,
)

NUM_CLASSES = 10
SUPERVISED_DEVICE = "Topcon_Maestro2"
UNLABELED_DEVICES = ["Heidelberg_Spectralis", "Zeiss_Cirrus", "Topcon_Maestro2_unlabeled"]


# ─────────────────────── utils ────────────────────────────────────────────────

def pick_device(opt):
    if opt != "auto":
        return torch.device(opt)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_amp(amp_opt, device):
    if amp_opt == "off" or device.type != "cuda":
        return False, None, False
    bf16_ok = torch.cuda.is_bf16_supported()
    if amp_opt == "bf16" or (amp_opt == "auto" and bf16_ok):
        return True, torch.bfloat16, False
    return True, torch.float16, True


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(ckpt_path, channels, img_size, device):
    side = ckpt_path + ".arch.json"
    if os.path.exists(side):
        a = json.load(open(side))
        channels = a.get("channels", channels)
        img_size  = a.get("img_size", img_size)
    strides = tuple(2 for _ in range(len(channels) - 1))
    model = UNet(
        spatial_dims=2, in_channels=1, out_channels=NUM_CLASSES,
        channels=tuple(channels), strides=strides, num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model, channels, img_size


# ─────────────────────── dados ────────────────────────────────────────────────

def list_supervised(root: Path):
    items = []
    for status in ["Diseased", "Healthy"]:
        for img in sorted(glob.glob(str(root / SUPERVISED_DEVICE / status / "*-image.png"))):
            mask = img.replace("-image.png", "-mask.png")
            if os.path.exists(mask):
                items.append({"image": img, "label": mask})
    return items


def list_unlabeled_images(root: Path):
    """
    Genérico: TODA imagem sem máscara-par, em QUALQUER subpasta de `root`.
    Pega Spectralis/Cirrus/Maestro2_unlabeled E TAMBÉM **wide-field** e qualquer
    outro domínio não rotulado que o servidor inclua — o gargalo do WideField era
    a lista hardcoded que ignorava o protocolo wide-field.
    """
    imgs = []
    for p in sorted(root.rglob("*-image.png")):
        mask = Path(str(p)[:-len("-image.png")] + "-mask.png")
        if mask.exists():
            continue  # rotulada (supervisionada) — pula
        rel = p.relative_to(root)
        vendor = rel.parts[0] if len(rel.parts) > 1 else "root"
        imgs.append((vendor, str(p)))
    return imgs


def build_transforms(img_size, aug, train):
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


# ─────────────────────── fase 1: pseudo-labels ────────────────────────────────

@torch.no_grad()
def generate_pseudo_labels(model, unlabeled_imgs, img_size, device, conf_threshold,
                            amp_enabled, amp_dtype, pseudo_dir):
    """
    Roda o modelo teacher em cada imagem não rotulada.
    Salva pseudo-máscara (uint8, valores 0-9) apenas se confiança >= threshold.
    Retorna lista de {"image": path, "label": pseudo_mask_path} para treino.
    """
    model.eval()
    pre = Compose([
        LoadImage(image_only=True, reverse_indexing=False),
        EnsureChannelFirst(), ScaleIntensity(), Resize((img_size, img_size)),
    ])
    os.makedirs(pseudo_dir, exist_ok=True)
    ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else nullcontext()

    accepted, total_by_vendor = [], {}
    print(f"\n[PSEUDO] gerando pseudo-labels (threshold={conf_threshold}) em {len(unlabeled_imgs)} imagens...")

    for vendor, img_path in unlabeled_imgs:
        total_by_vendor[vendor] = total_by_vendor.get(vendor, 0) + 1
        img_t = torch.as_tensor(pre(img_path)).unsqueeze(0).to(device)
        with ctx:
            logits = model(img_t)                              # (1, C, H, W)
        probs  = torch.softmax(logits.float(), dim=1)          # (1, C, H, W)
        conf   = probs.max(dim=1).values.mean().item()         # escalar: mean max-prob
        if conf < conf_threshold:
            continue
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)  # (H, W), values 0-9

        stem = os.path.splitext(os.path.basename(img_path))[0]
        pseudo_path = os.path.join(pseudo_dir, f"{stem}-pseudo_mask.png")
        cv2.imwrite(pseudo_path, pred)
        accepted.append({"image": img_path, "label": pseudo_path, "_vendor": vendor})

    accepted_by_vendor = {}
    for item in accepted:
        v = item["_vendor"]
        accepted_by_vendor[v] = accepted_by_vendor.get(v, 0) + 1

    print(f"[PSEUDO] aceitos {len(accepted)}/{len(unlabeled_imgs)} (conf>={conf_threshold}):")
    for v in sorted(total_by_vendor):  # domínios descobertos (inclui wide-field se houver)
        tot = total_by_vendor.get(v, 0)
        acc = accepted_by_vendor.get(v, 0)
        print(f"         {v}: {acc}/{tot} ({100*acc/max(1,tot):.1f}%)")
    return accepted


# ─────────────────────── val ──────────────────────────────────────────────────

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
        pred = logits.argmax(dim=1)
        pred_oh = torch.nn.functional.one_hot(pred, NUM_CLASSES).permute(0, 3, 1, 2).float()
        y_oh   = torch.nn.functional.one_hot(y.squeeze(1), NUM_CLASSES).permute(0, 3, 1, 2).float()
        metric(y_pred=pred_oh, y=y_oh)
    return float(metric.aggregate().item())


# ─────────────────────── main ─────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",       required=True)
    ap.add_argument("--teacher",         required=True,  help="checkpoint do modelo supervisionado (round2)")
    ap.add_argument("--out",             default="models/round3_semi")
    ap.add_argument("--pseudo_dir",      default="",     help="onde salvar pseudo-masks (default: <out>/pseudo)")
    ap.add_argument("--epochs",          type=int,   default=150)
    ap.add_argument("--batch_size",      type=int,   default=16)
    ap.add_argument("--img_size",        type=int,   default=384)
    ap.add_argument("--lr",              type=float, default=3e-4)
    ap.add_argument("--channels",        default="32,64,128,256")
    ap.add_argument("--aug",             choices=["none","basic","strong","widefield","widefield2"], default="strong")
    ap.add_argument("--val_frac",        type=float, default=0.15)
    ap.add_argument("--conf_threshold",  type=float, default=0.85,
                    help="mean max-prob mínima p/ aceitar pseudo-label (0-1)")
    ap.add_argument("--semi_weight",     type=float, default=0.5,
                    help="peso relativo do pseudo-label no loss (1.0 = igual ao supervisionado)")
    ap.add_argument("--device",          default="auto", choices=["auto","cuda","mps","cpu"])
    ap.add_argument("--amp",             default="auto", choices=["auto","bf16","fp16","off"])
    ap.add_argument("--workers",         type=int,   default=4)
    ap.add_argument("--cache_rate",      type=float, default=1.0)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--time_budget_min", type=float, default=0)
    ap.add_argument("--skip_pseudo",     action="store_true",
                    help="pula a geração de pseudo-labels (reutiliza --pseudo_dir existente)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    amp_enabled, amp_dtype, use_scaler = pick_amp(args.amp, device)
    channels = [int(c) for c in args.channels.split(",")]
    pseudo_dir = args.pseudo_dir or os.path.join(args.out, "pseudo")
    os.makedirs(args.out, exist_ok=True)

    print("=" * 60)
    print(" F2 — SEMI-SUPERVISÃO POR PSEUDO-LABELS")
    print(f" device     : {device.type}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f" teacher    : {args.teacher}")
    print(f" AMP        : enabled={amp_enabled} dtype={amp_dtype}")
    print(f" arch       : UNet{tuple(channels)} @ {args.img_size}px")
    print(f" conf thr   : {args.conf_threshold}  semi_weight={args.semi_weight}")
    print(f" epochs     : {args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print("=" * 60)

    # ── carrega teacher e lê arch do sidecar ──────────────────────────────────
    model, channels, img_size = load_model(args.teacher, channels, args.img_size, device)
    print(f"[OK] teacher carregado: UNet{tuple(channels)} img_size={img_size}")

    # ── fase 1: gerar pseudo-labels ───────────────────────────────────────────
    data_root = Path(args.data_root)
    if not args.skip_pseudo:
        unlabeled_imgs = list_unlabeled_images(data_root)
        pseudo_items = generate_pseudo_labels(
            model, unlabeled_imgs, img_size, device,
            args.conf_threshold, amp_enabled, amp_dtype, pseudo_dir,
        )
    else:
        pseudo_items = []
        for p in glob.glob(os.path.join(pseudo_dir, "*-pseudo_mask.png")):
            # recupera imagem original em qualquer subpasta (busca genérica)
            stem = os.path.basename(p).replace("-pseudo_mask.png", "-image.png")
            matches = list(Path(data_root).rglob(stem))
            if matches:
                pseudo_items.append({"image": str(matches[0]), "label": p})
        print(f"[PSEUDO] reutilizando {len(pseudo_items)} pseudo-masks de {pseudo_dir}")

    # ── fase 2: re-treino ─────────────────────────────────────────────────────
    supervised = list_supervised(data_root)
    if not supervised:
        sys.exit(f"[ERRO] nenhum par image/mask em {data_root}/{SUPERVISED_DEVICE}")

    random.shuffle(supervised)
    n_val = max(1, int(len(supervised) * args.val_frac))
    val_items, train_sup = supervised[:n_val], supervised[n_val:]

    # remove _vendor key antes de passar ao dataset (campo extra não tolerado pelo MONAI)
    pseudo_train = [{"image": it["image"], "label": it["label"]} for it in pseudo_items]

    print(f"\n[TRAIN] supervisionado: {len(train_sup)} treino / {len(val_items)} val")
    print(f"[TRAIN] pseudo-rotulados: {len(pseudo_train)} amostras")

    tf_train = build_transforms(img_size, args.aug, True)
    tf_val   = build_transforms(img_size, "none",  False)

    sup_ds   = CacheDataset(train_sup,    tf_train, cache_rate=args.cache_rate, num_workers=args.workers)
    pseudo_ds= CacheDataset(pseudo_train, tf_train, cache_rate=args.cache_rate, num_workers=args.workers)
    val_ds   = CacheDataset(val_items,    tf_val,   cache_rate=args.cache_rate, num_workers=args.workers)

    sup_loader    = DataLoader(sup_ds,    batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, drop_last=True)
    pseudo_loader = DataLoader(pseudo_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, drop_last=False) if pseudo_ds else None
    val_loader    = DataLoader(val_ds,    batch_size=args.batch_size, num_workers=args.workers)

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    opt     = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler  = torch.amp.GradScaler("cuda", enabled=use_scaler)

    ckpt_path = os.path.join(args.out, "unet_maestro2_semi.pth")
    with open(ckpt_path + ".arch.json", "w") as f:
        json.dump({"channels": channels, "img_size": img_size, "num_classes": NUM_CLASSES}, f)

    best_val, history = -1.0, []
    t0 = time.time()
    pseudo_iter = iter(pseudo_loader) if pseudo_loader else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_batches = 0

        for batch in sup_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device).long()
            opt.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_enabled else nullcontext()
            with ctx:
                loss = loss_fn(model(x), y)
            if use_scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            ep_loss += loss.item(); n_batches += 1

            # intercala um batch de pseudo-rotulados com peso semi_weight
            if pseudo_iter and args.semi_weight > 0:
                try:
                    pb = next(pseudo_iter)
                except StopIteration:
                    pseudo_iter = iter(pseudo_loader)
                    pb = next(pseudo_iter)
                xp = pb["image"].to(device)
                yp = pb["label"].to(device).long()
                opt.zero_grad(set_to_none=True)
                with ctx:
                    semi_loss = loss_fn(model(xp), yp) * args.semi_weight
                if use_scaler:
                    scaler.scale(semi_loss).backward(); scaler.step(opt); scaler.update()
                else:
                    semi_loss.backward(); opt.step()
                ep_loss += semi_loss.item(); n_batches += 1

        sched.step()
        val_dice = evaluate(model, val_loader, device, amp_enabled, amp_dtype)
        avg_loss = ep_loss / max(1, n_batches)
        history.append({"epoch": epoch, "loss": avg_loss, "val_dice": val_dice})

        flag = ""
        if val_dice > best_val:
            best_val = val_dice
            torch.save(model.state_dict(), ckpt_path)
            flag = "  <- best"

        elapsed = (time.time() - t0) / 60.0
        print(f"[{epoch:3d}/{args.epochs}] loss={avg_loss:.4f} val_dice={val_dice:.4f}{flag}  ({elapsed:.1f}min)")

        if args.time_budget_min and elapsed >= args.time_budget_min:
            print(f"[INFO] teto de tempo ({args.time_budget_min}min) atingido — parando na época {epoch}.")
            break

    report = {
        "best_val_dice": best_val,
        "epochs": args.epochs,
        "channels": channels,
        "img_size": img_size,
        "aug": args.aug,
        "amp": str(amp_dtype),
        "device": device.type,
        "conf_threshold": args.conf_threshold,
        "pseudo_accepted": len(pseudo_train),
        "history": history,
    }
    with open(os.path.join(args.out, "train_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[OK] best val_dice={best_val:.4f}  checkpoint={ckpt_path}")
    print(f"     pseudo-labels aceitos: {len(pseudo_train)}")


if __name__ == "__main__":
    main()
