#!/usr/bin/env python3
"""
Relatório de confiança cross-vendor.

Roda o modelo em todas as imagens não rotuladas (Spectralis, Cirrus, Maestro2_unlabeled)
e reporta a distribuição de confiança (mean max-softmax) por vendor.
Alta confiança = modelo generaliza bem para aquele domínio.
Salva resultados em results/crossvendor_confidence.json.

Uso:
  python scripts/eval_crossvendor.py \\
    --data_root data/starter_kit/app_ingestion/input_data/train \\
    --model     models/round3_semi/unet_maestro2_semi.pth \\
    --out       results/crossvendor_confidence.json
"""
import argparse, glob, json, os
from pathlib import Path

import numpy as np
import torch
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize

NUM_CLASSES = 10
UNLABELED_DEVICES = ["Heidelberg_Spectralis", "Zeiss_Cirrus", "Topcon_Maestro2_unlabeled"]


def load_model(ckpt, device):
    side = ckpt + ".arch.json"
    channels, img_size = [16, 32, 64, 128], 256
    if os.path.exists(side):
        a = json.load(open(side))
        channels = a.get("channels", channels)
        img_size  = a.get("img_size", img_size)
    strides = tuple(2 for _ in range(len(channels) - 1))
    model = UNet(spatial_dims=2, in_channels=1, out_channels=NUM_CLASSES,
                 channels=tuple(channels), strides=strides, num_res_units=2).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, img_size


@torch.no_grad()
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, img_size = load_model(args.model, device)
    pre = Compose([LoadImage(image_only=True, reverse_indexing=False),
                   EnsureChannelFirst(), ScaleIntensity(), Resize((img_size, img_size))])

    print(f"[CV] modelo: {args.model}  device={device.type}  img_size={img_size}")
    results = {}

    for vendor in UNLABELED_DEVICES:
        confs = []
        imgs = []
        for status in ["Diseased", "Healthy"]:
            imgs += sorted(glob.glob(
                str(Path(args.data_root) / vendor / status / "*-image.png")))
        if not imgs:
            print(f"[CV] {vendor}: nenhuma imagem encontrada")
            continue
        for p in imgs:
            img_t = torch.as_tensor(pre(p)).unsqueeze(0).to(device)
            probs = torch.softmax(model(img_t).float(), dim=1)
            conf  = probs.max(dim=1).values.mean().item()
            confs.append(conf)
        confs = np.array(confs)
        results[vendor] = {
            "n": len(confs),
            "mean_conf": float(confs.mean()),
            "median_conf": float(np.median(confs)),
            "pct_above_85": float((confs >= 0.85).mean() * 100),
            "pct_above_90": float((confs >= 0.90).mean() * 100),
            "pct_above_95": float((confs >= 0.95).mean() * 100),
        }
        print(f"[CV] {vendor}: n={len(confs)}  mean={confs.mean():.4f}  "
              f"≥0.85={results[vendor]['pct_above_85']:.1f}%  "
              f"≥0.90={results[vendor]['pct_above_90']:.1f}%  "
              f"≥0.95={results[vendor]['pct_above_95']:.1f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[CV] salvo em {args.out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model",     required=True)
    ap.add_argument("--out",       default="results/crossvendor_confidence.json")
    run(ap.parse_args())
