#!/usr/bin/env python3
"""
Avalia um ENSEMBLE (média de softmax de N modelos) no proxy de wide-field.
Cada modelo prediz no seu próprio img_size, as probs são levadas à resolução nativa e somadas;
argmax no fim. Suporta TTA por modelo. Inferência-pura (sem retreino).

Uso:
  python scripts/eval_ensemble_proxy.py --models m1.pth m2.pth m3.pth --warp cosine --amp 0.30 [--tta]
"""
import argparse, glob, json, os, sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensity, Resize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_proxy_widefield import warp_image, load_model  # reusa  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "data/starter_kit/app_scoring/program"))
from metrics import compute_image_score  # noqa: E402

KIT = os.path.join(ROOT, "data/starter_kit")
VAL_IMAGES = os.path.join(KIT, "app_ingestion/input_data/val/images")
VAL_MASKS = os.path.join(KIT, "app_scoring/input/ref/val/masks")


def model_probs(model, img_u8, img_size, device, tta, out_hw):
    """Retorna softmax [1,C,H,W] na resolução nativa (out_hw)."""
    pre = Compose([EnsureChannelFirst(channel_dim="no_channel"), ScaleIntensity(), Resize((img_size, img_size))])
    x = torch.as_tensor(pre(img_u8.astype(np.float32))).unsqueeze(0).to(device)
    with torch.no_grad():
        if not tta:
            probs = torch.softmax(model(x).float(), 1)
        else:
            accum, n = None, 0
            H, W = x.shape[-2], x.shape[-1]
            for scale in (1.0, 1.15):
                xs = x if scale == 1.0 else F.interpolate(
                    x, size=(int(round(H*scale/8))*8, int(round(W*scale/8))*8), mode="bilinear", align_corners=False)
                for flip in (False, True):
                    v = torch.flip(xs, dims=[3]) if flip else xs
                    p = torch.softmax(model(v).float(), 1)
                    if flip:
                        p = torch.flip(p, dims=[3])
                    if p.shape[-2:] != x.shape[-2:]:
                        p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
                    accum = p if accum is None else accum + p; n += 1
            probs = accum / n
    return F.interpolate(probs, size=out_hw, mode="bilinear", align_corners=False)


def ensemble_pred(models_info, img_u8, device, tta):
    h, w = img_u8.shape[:2]
    accum = None
    for m, isz in models_info:
        p = model_probs(m, img_u8, isz, device, tta, (h, w))
        accum = p if accum is None else accum + p
    return torch.argmax(accum, 1)[0].cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--warp", choices=["cosine", "radial"], default="cosine")
    ap.add_argument("--amp", type=float, default=0.30)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    models_info = []
    for mp in args.models:
        m, isz = load_model(mp, device)
        models_info.append((m, isz))
    print(f"[INFO] ensemble de {len(models_info)} modelos  device={device.type} warp={args.warp} tta={args.tta}")

    masks = sorted(glob.glob(os.path.join(VAL_MASKS, "*-mask.png")))
    plain, warped = [], []
    for mpth in masks:
        name = os.path.basename(mpth).replace("-mask.png", "-image.png")
        img = cv2.imread(os.path.join(VAL_IMAGES, name), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(mpth, cv2.IMREAD_GRAYSCALE)
        if img is None or gt is None:
            continue
        p0 = ensemble_pred(models_info, img, device, args.tta)
        plain.append(compute_image_score(p0, gt)[0])
        iw = warp_image(img, args.amp, "image", args.warp); gw = warp_image(gt, args.amp, "mask", args.warp)
        pw = ensemble_pred(models_info, iw, device, args.tta)
        warped.append(compute_image_score(pw, gw)[0])
    print(f"  plain={np.mean(plain):.4f}  {args.warp}={np.mean(warped):.4f}")


if __name__ == "__main__":
    main()
