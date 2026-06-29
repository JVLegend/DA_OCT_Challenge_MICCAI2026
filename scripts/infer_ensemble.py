#!/usr/bin/env python3
"""
Inferência por ENSEMBLE: média de softmax de N modelos (cada um na sua resolução) + TTA,
depois refine + resolução nativa. Validado offline (proxy): multi-res {256,384,512}+TTA
melhora Mácula +0.025 e WideField +0.022 vs o melhor único.

Uso:
  python scripts/infer_ensemble.py --models m0.pth m1.pth m2.pth \
      --input_dir <dir> --output_dir <dir> [--refine --native_size --tta]

Reaproveita discover_images / out_name / refine_boundaries / load_arch do infer_daoct.
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from infer_daoct import discover_images, out_name, refine_boundaries, NUM_CLASSES  # noqa: E402


def load_one(model_path, device):
    side = model_path + ".arch.json"
    a = json.load(open(side)) if os.path.exists(side) else {"channels": [16, 32, 64, 128], "img_size": 256}
    ch, isz = a.get("channels", [16, 32, 64, 128]), a.get("img_size", 256)
    strides = tuple(2 for _ in range(len(ch) - 1))
    m = UNet(spatial_dims=2, in_channels=1, out_channels=NUM_CLASSES,
             channels=tuple(ch), strides=strides, num_res_units=2).to(device)
    m.load_state_dict(torch.load(model_path, map_location=device)); m.eval()
    return m, isz


def model_probs(model, x_native, isz, device, tta, out_hw):
    """x_native: tensor [1,1,Hn,Wn] (intensidade já escalada). Retorna softmax [1,C,*out_hw]."""
    x = F.interpolate(x_native, size=(isz, isz), mode="bilinear", align_corners=False)
    with torch.no_grad():
        if not tta:
            probs = torch.softmax(model(x).float(), 1)
        else:
            accum, n = None, 0
            for scale in (1.0, 1.15):
                xs = x if scale == 1.0 else F.interpolate(
                    x, size=(int(round(isz*scale/8))*8, int(round(isz*scale/8))*8), mode="bilinear", align_corners=False)
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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    models = [load_one(p, device) for p in args.models]
    print(f"[INFO] ensemble de {len(models)} modelos: {[(tuple_isz[1]) for tuple_isz in models]}px  device={device.type}"
          + (" +refine" if args.refine else "") + (" +native" if args.native_size else "") + (" +tta" if args.tta else ""))

    pre = Compose([LoadImage(image_only=True, reverse_indexing=False), EnsureChannelFirst(), ScaleIntensity()])
    images = discover_images(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    for img_path in images:
        x_native = torch.as_tensor(pre(img_path)).unsqueeze(0).to(device)  # [1,1,Hn,Wn]
        oh, ow = x_native.shape[-2], x_native.shape[-1]
        # resolução de ensemble: maior img_size dos modelos (boa para todos)
        ens_sz = max(isz for _, isz in models)
        accum = None
        for m, isz in models:
            p = model_probs(m, x_native, isz, device, args.tta, (ens_sz, ens_sz))
            accum = p if accum is None else accum + p
        pred = torch.argmax(accum, 1)[0].cpu().numpy().astype(np.uint8)
        if args.refine:
            pred = refine_boundaries(pred)
        if args.native_size and (oh, ow) != pred.shape:
            pred = cv2.resize(pred, (ow, oh), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(args.output_dir, out_name(img_path)), pred)

    print(f"[INFO] ensemble: {len(images)} máscaras em {args.output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--native_size", action="store_true")
    ap.add_argument("--tta", action="store_true")
    main(ap.parse_args())
