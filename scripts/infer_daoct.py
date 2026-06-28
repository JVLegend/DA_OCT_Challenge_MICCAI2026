#!/usr/bin/env python3
"""
Inferência arch-aware e robusta a path.

- Reconstrói o UNet com a arquitetura salva no sidecar `<model>.arch.json` (ou via --channels),
  então funciona com qualquer tamanho de modelo (round 1 baseline OU archs maiores).
- Descoberta de imagens RECURSIVA e tolerante: nunca volta 0 silenciosamente. Isso evita o gotcha
  do fórum (#120): "rodou nas 173 mas deu Failed" porque o glob achou 0 imagens no path do servidor.
- Refinamento de bordas (--refine): suaviza os limites de camada horizontalmente via superfícies
  gaussianas, reduzindo MASD (τ=0.02 é severo — cada pixel de borda conta muito).
- Resolução nativa (--native_size): salva a máscara no tamanho original da imagem em vez de
  manter o tamanho do modelo (evita upscale NEAREST pelo scorer).

Uso:
  python scripts/infer_daoct.py --input_dir <dir> --output_dir <dir> --model_path <ckpt.pth>
  python scripts/infer_daoct.py ... --refine --native_size
"""
import argparse, glob, json, os
import numpy as np
import torch
import cv2
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, Resize

NUM_CLASSES = 10
DEFAULT_CHANNELS = [16, 32, 64, 128]


def load_arch(model_path, cli_channels, cli_img):
    side = model_path + ".arch.json"
    if os.path.exists(side):
        a = json.load(open(side))
        return a.get("channels", DEFAULT_CHANNELS), a.get("img_size", cli_img), a.get("num_classes", NUM_CLASSES)
    ch = [int(c) for c in cli_channels.split(",")] if cli_channels else DEFAULT_CHANNELS
    return ch, cli_img, NUM_CLASSES


def discover_images(input_dir):
    """Recursivo e tolerante. Tenta padrões em ordem; loga o que achou."""
    pats = ["**/*-image.png", "**/*_image.png", "**/*.png", "**/*.jpg", "**/*.jpeg", "**/*.tif", "**/*.tiff"]
    for p in pats:
        hits = sorted(glob.glob(os.path.join(input_dir, p), recursive=True))
        hits = [h for h in hits if "-mask" not in os.path.basename(h).lower()
                and "-pseudo_mask" not in os.path.basename(h).lower()]
        if hits:
            print(f"[INFO] discovery: {len(hits)} imagens via padrão '{p}' em {input_dir}")
            return hits
    print(f"[WARN] NENHUMA imagem encontrada em {input_dir} (tentei {pats})")
    return []


def out_name(img_path):
    base = os.path.basename(img_path)
    for tok in ["-image.png", "_image.png"]:
        if tok in base:
            return base.replace(tok, "-mask.png")
    stem = os.path.splitext(base)[0]
    return f"{stem}-mask.png"


def refine_boundaries(pred: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """
    Suaviza os limites das camadas retinianas horizontalmente.

    Abordagem: codifica o argmax em one-hot, aplica Gaussian 1D horizontal em cada
    canal de classe, e recomputa o argmax. Isso suaviza zigzag nas bordas sem
    deslocar os limites para dentro das classes — ao contrário de métodos baseados
    em posição média, que produzem artefatos.

    σ=2.5 → janela efetiva de ~5px horizontal (suficiente para remover ruído de 1-2px
    sem borrar camadas finas).
    """
    from scipy.ndimage import gaussian_filter

    # One-hot: (H, W, C)
    onehot = (pred[:, :, None] == np.arange(NUM_CLASSES)[None, None, :]).astype(np.float32)
    # Gaussian horizontal (sigma só em W, eixo 1)
    smoothed = gaussian_filter(onehot, sigma=(0, sigma, 0))
    return smoothed.argmax(axis=-1).astype(np.uint8)


def tta_predict(model, x):
    """TTA: média de softmax sobre h-flip + 2 escalas (1.0, 1.15). SEM v-flip — inverteria a
    ordem vertical das camadas da retina. Validado offline (cosine/WideField +0.045)."""
    import torch.nn.functional as F
    H, W = x.shape[-2], x.shape[-1]
    accum, n = None, 0
    for scale in (1.0, 1.15):
        if scale == 1.0:
            xs = x
        else:  # múltiplo de 8 (UNet com 3 downsamples exige)
            nh, nw = int(round(H * scale / 8)) * 8, int(round(W * scale / 8)) * 8
            xs = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
        for flip in (False, True):
            v = torch.flip(xs, dims=[3]) if flip else xs
            p = torch.softmax(model(v).float(), 1)
            if flip:
                p = torch.flip(p, dims=[3])
            if p.shape[-2:] != x.shape[-2:]:
                p = F.interpolate(p, size=x.shape[-2:], mode="bilinear", align_corners=False)
            accum = p if accum is None else accum + p
            n += 1
    return accum / n


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if getattr(torch.backends, "mps", None)
                                and torch.backends.mps.is_available() else "cpu"))
    channels, img_size, num_classes = load_arch(args.model_path, args.channels, args.img_size)
    strides = tuple(2 for _ in range(len(channels) - 1))
    model = UNet(spatial_dims=2, in_channels=1, out_channels=num_classes,
                 channels=tuple(channels), strides=strides, num_res_units=2).to(device)
    print(f"[INFO] device={device.type} arch=UNet{tuple(channels)} img_size={img_size}"
          + (" +refine" if args.refine else "") + (" +native" if args.native_size else "")
          + (" +tta" if args.tta else ""))
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    pre = Compose([LoadImage(image_only=True, reverse_indexing=False),
                   EnsureChannelFirst(), ScaleIntensity(), Resize((img_size, img_size))])
    pre_native = Compose([LoadImage(image_only=True, reverse_indexing=False),
                          EnsureChannelFirst(), ScaleIntensity()])

    images = discover_images(args.input_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    for img_path in images:
        # Load at native size to record original dims
        native = pre_native(img_path)
        orig_h, orig_w = native.shape[-2], native.shape[-1]

        img = torch.as_tensor(pre(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = tta_predict(model, img) if args.tta else torch.softmax(model(img).float(), 1)
            pred = torch.argmax(probs, dim=1)[0].cpu().numpy().astype(np.uint8)

        if args.refine:
            pred = refine_boundaries(pred)

        if args.native_size and (orig_h != img_size or orig_w != img_size):
            pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(args.output_dir, out_name(img_path)), pred)

    print(f"[INFO] inferência completa: {len(images)} máscaras em {args.output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--channels", default="", help="ex '32,64,128,256' (ignorado se houver sidecar .arch.json)")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--refine", action="store_true", help="suaviza limites de camada (melhora MASD)")
    ap.add_argument("--native_size", action="store_true", help="salva máscara na resolução original da imagem")
    ap.add_argument("--tta", action="store_true", help="test-time aug: h-flip + 2 escalas (média de softmax)")
    main(ap.parse_args())
