#!/usr/bin/env python3
"""
Proxy OFFLINE de wide-field: deforma a val de Mácula com geometria sintética
(mimetiza o FOV 12x9mm do wide-field, que curva muito mais que a Mácula 6x6) e mede
quanto o modelo degrada. Não é wide-field real, mas dá um GRADIENTE pra ajustar a
augmentation de generalização geométrica sem gastar bala.

Dois tipos de warp (use ambos p/ não viciar a aug num só):
  --warp cosine  : curva vertical global (coluna a coluna) — curvatura suave do campo largo
  --warp radial  : distorção barrel 2D (afasta do centro) — geometria bem diferente

Reporta image_score médio (=média das 10 classes de 0.5·(Dice + exp(-MASD/τ))):
  sem warp (sanity) vs com warp (a QUEDA mede a fragilidade geométrica).

Uso:
  python scripts/eval_proxy_widefield.py --model_path <ckpt.pth> --warp cosine --amp 0.30
"""
import argparse, glob, json, os, sys
import numpy as np
import cv2
import torch
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensity, Resize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data/starter_kit/app_scoring/program"))
from metrics import compute_image_score  # noqa: E402

KIT = os.path.join(ROOT, "data/starter_kit")
VAL_IMAGES = os.path.join(KIT, "app_ingestion/input_data/val/images")
VAL_MASKS = os.path.join(KIT, "app_scoring/input/ref/val/masks")


def warp_image(img, amp, mode, warp):
    """Mesma transformação geométrica p/ imagem (linear) e máscara (nearest)."""
    h, w = img.shape[:2]
    interp = cv2.INTER_NEAREST if mode == "mask" else cv2.INTER_LINEAR
    if warp == "cosine":
        xs = np.arange(w)
        dy = (amp * h) * np.cos(2 * np.pi * xs / max(1, w - 1))
        mapx = np.tile(xs, (h, 1)).astype(np.float32)
        mapy = (np.arange(h)[:, None] - dy[None, :]).astype(np.float32)
    elif warp == "radial":  # barrel 2D — afasta do centro
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        nx, ny = (xx - cx) / max(1.0, cx), (yy - cy) / max(1.0, cy)
        factor = 1.0 + amp * (nx * nx + ny * ny)
        mapx = (cx + (xx - cx) * factor).astype(np.float32)
        mapy = (cy + (yy - cy) * factor).astype(np.float32)
    else:
        raise ValueError(warp)
    return cv2.remap(img, mapx, mapy, interpolation=interp, borderMode=cv2.BORDER_REFLECT)


def load_model(model_path, device):
    side = model_path + ".arch.json"
    a = json.load(open(side)) if os.path.exists(side) else {"channels": [16, 32, 64, 128], "img_size": 256}
    ch, img_size = a["channels"], a["img_size"]
    strides = tuple(2 for _ in range(len(ch) - 1))
    m = UNet(spatial_dims=2, in_channels=1, out_channels=10, channels=tuple(ch),
             strides=strides, num_res_units=2).to(device)
    m.load_state_dict(torch.load(model_path, map_location=device)); m.eval()
    return m, img_size


def predict(model, img_u8, img_size, device):
    pre = Compose([EnsureChannelFirst(channel_dim="no_channel"), ScaleIntensity(), Resize((img_size, img_size))])
    x = torch.as_tensor(pre(img_u8.astype(np.float32))).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.argmax(model(x), 1)[0].cpu().numpy().astype(np.uint8)
    return cv2.resize(pred, (img_u8.shape[1], img_u8.shape[0]), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--warp", choices=["cosine", "radial"], default="cosine")
    ap.add_argument("--amp", type=float, default=0.30)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model, img_size = load_model(args.model_path, device)
    print(f"[INFO] device={device.type} img_size={img_size} warp={args.warp} amp={args.amp}")

    masks = sorted(glob.glob(os.path.join(VAL_MASKS, "*-mask.png")))
    if args.limit:
        masks = masks[: args.limit]

    plain, warped = [], []
    for mp in masks:
        name = os.path.basename(mp).replace("-mask.png", "-image.png")
        ip = os.path.join(VAL_IMAGES, name)
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        p0 = predict(model, img, img_size, device)
        s0, _, _, _ = compute_image_score(cv2.resize(p0, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST), gt)
        plain.append(s0)
        img_w = warp_image(img, args.amp, "image", args.warp)
        gt_w = warp_image(gt, args.amp, "mask", args.warp)
        pw = predict(model, img_w, img_size, device)
        sw, _, _, _ = compute_image_score(cv2.resize(pw, (gt_w.shape[1], gt_w.shape[0]), interpolation=cv2.INTER_NEAREST), gt_w)
        warped.append(sw)

    p, w = float(np.mean(plain)), float(np.mean(warped))
    print(f"\n[PROXY] n={len(plain)} warp={args.warp}")
    print(f"  image_score SEM warp : {p:.4f}")
    print(f"  image_score COM warp : {w:.4f}")
    print(f"  QUEDA: {p - w:.4f} ({100*(p-w)/max(1e-9,p):.1f}%)")


if __name__ == "__main__":
    main()
