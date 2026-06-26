"""
DA-OCT Challenge — entrypoint principal do container de submissão.

Chamado por entrypoint.sh:
  python3 main.py <input_data> <output_dir> <submission_dir>

Fluxo:
  1. Detecta GPU, calcula orçamento de tempo.
  2. Localiza dados de treino e imagens de inferência robustamente.
  3. Supervised (warm-start no checkpoint pré-treinado, ou treino do zero).
  4. Semi-supervisão por pseudo-labels nos não rotulados.
  5. Inferência com refinamento de bordas + resolução nativa.
"""
import json, os, subprocess, sys, time
from pathlib import Path

SEED = 42
TOTAL_BUDGET_MIN = 95   # 95 min treino + ~5 min infer = bem dentro das 2h


def log(msg):
    print(f"[MAIN] {msg}", flush=True)


def py():
    return sys.executable


def find_training_root(input_dir: str) -> str:
    for cand in [
        os.path.join(input_dir, "train"),
        os.path.join(input_dir, "data", "train"),
        input_dir,
    ]:
        if os.path.isdir(os.path.join(cand, "Topcon_Maestro2")):
            return cand
    return input_dir


def find_inference_dir(input_dir: str) -> str:
    for cand in [
        os.path.join(input_dir, "ref"),          # server: /app/input/ref
        os.path.join(input_dir, "val", "images"),
        os.path.join(input_dir, "val"),
        os.path.join(input_dir, "testing_data"),
        input_dir,
    ]:
        if os.path.isdir(cand):
            imgs = list(Path(cand).rglob("*-image.png"))
            if imgs:
                return cand
    return input_dir


def find_pretrained(submission_dir: str):
    for name in [
        "checkpoints/unet_maestro2_semi.pth",
        "checkpoints/unet_maestro2.pth",
        "unet_maestro2_semi.pth",
        "unet_maestro2.pth",
    ]:
        p = os.path.join(submission_dir, name)
        if os.path.exists(p):
            return p
    return None


def run(cmd, **kw):
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    input_dir     = os.path.abspath(sys.argv[1])
    output_dir    = os.path.abspath(sys.argv[2])
    sub_dir       = os.path.abspath(sys.argv[3])
    os.chdir(sub_dir)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    log(f"input_dir    = {input_dir}")
    log(f"output_dir   = {output_dir}")
    log(f"sub_dir      = {sub_dir}")

    data_root  = find_training_root(input_dir)
    infer_dir  = find_inference_dir(input_dir)
    ckpt_out   = os.path.join(sub_dir, "checkpoints", "unet_maestro2_semi.pth")
    pseudo_dir = os.path.join(sub_dir, "checkpoints", "pseudo")

    log(f"data_root    = {data_root}")
    log(f"infer_dir    = {infer_dir}")

    pretrained = find_pretrained(sub_dir)
    warm_start = pretrained and pretrained != ckpt_out

    def elapsed_min():
        return (time.time() - t0) / 60.0

    def remaining_min():
        return max(0, TOTAL_BUDGET_MIN - elapsed_min())

    # ── 1. Treino supervisionado ──────────────────────────────────────────────
    if warm_start:
        log(f"Warm-start encontrado: {pretrained} → fine-tune supervisionado")
        # Renomeia o checkpoint pré-treinado para não sobrescrever durante o treino
        pretrained_backup = os.path.join(sub_dir, "checkpoints", "pretrained.pth")
        if not os.path.exists(pretrained_backup):
            import shutil; shutil.copy2(pretrained, pretrained_backup)
            arch_src = pretrained + ".arch.json"
            if os.path.exists(arch_src):
                shutil.copy2(arch_src, pretrained_backup + ".arch.json")
        epochs_sup  = 60
        budget_sup  = min(40, remaining_min() * 0.45)
    else:
        log("Sem checkpoint pré-treinado — treino supervisionado completo")
        pretrained_backup = None
        epochs_sup  = 150
        budget_sup  = min(55, remaining_min() * 0.55)

    # Lê arch do sidecar (para manter mesma arch do pré-treinado)
    arch_file = (pretrained_backup or "") + ".arch.json" if pretrained_backup else None
    channels = "32,64,128,256"
    img_size  = 384
    if arch_file and os.path.exists(arch_file):
        a = json.load(open(arch_file))
        channels = ",".join(str(c) for c in a.get("channels", [32, 64, 128, 256]))
        img_size  = a.get("img_size", 384)

    cmd_sup = [
        py(), "train_daoct.py",
        "--data_root", data_root,
        "--out", "checkpoints",
        "--epochs", str(epochs_sup),
        "--batch_size", "16",
        "--img_size", str(img_size),
        "--channels", channels,
        "--amp", "auto",
        "--aug", "strong",
        "--workers", "4",
        "--seed", str(SEED),
        "--time_budget_min", f"{budget_sup:.1f}",
    ]
    if pretrained_backup:
        # train_daoct.py aceita --warm_start se implementado (fallback: treina do zero)
        pass
    run(cmd_sup, cwd=sub_dir)
    log(f"Supervisionado concluído em {elapsed_min():.1f} min")

    # ── 2. Semi-supervisão ────────────────────────────────────────────────────
    epochs_semi = 60 if warm_start else 80
    budget_semi = min(40, remaining_min() * 0.90)

    cmd_semi = [
        py(), "train_daoct_semi.py",
        "--data_root", data_root,
        "--teacher",   ckpt_out,
        "--out",       "checkpoints",
        "--pseudo_dir", pseudo_dir,
        "--epochs", str(epochs_semi),
        "--batch_size", "16",
        "--img_size", str(img_size),
        "--channels", channels,
        "--amp", "auto",
        "--aug", "strong",
        "--conf_threshold", "0.82",
        "--semi_weight", "0.5",
        "--workers", "4",
        "--seed", str(SEED),
        "--time_budget_min", f"{budget_semi:.1f}",
    ]
    run(cmd_semi, cwd=sub_dir)
    log(f"Semi-supervisão concluída em {elapsed_min():.1f} min")

    # ── 3. Inferência ─────────────────────────────────────────────────────────
    log(f"Inferindo em {infer_dir} → {output_dir}")
    run([
        py(), "infer_daoct.py",
        "--input_dir",  infer_dir,
        "--output_dir", output_dir,
        "--model_path", ckpt_out,
        "--refine",
        "--native_size",
    ], cwd=sub_dir)

    log(f"Concluído em {elapsed_min():.1f} min total.")


if __name__ == "__main__":
    main()
