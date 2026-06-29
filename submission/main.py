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
TOTAL_BUDGET_MIN = 100   # modelo maior (48-384) é ~2-3x mais lento/época; cap mantém total < 2h com folga


def log(msg):
    print(f"[MAIN] {msg}", flush=True)


def py():
    return sys.executable


def find_training_root(input_dir: str) -> str:
    # candidatos rápidos (cobre /app/input/data, .../data/train, etc.)
    for cand in [
        os.path.join(input_dir, "train"),
        os.path.join(input_dir, "data", "train"),
        os.path.join(input_dir, "data"),
        input_dir,
    ]:
        if os.path.isdir(os.path.join(cand, "Topcon_Maestro2")):
            return cand
    # fallback robusto: acha .../Topcon_Maestro2 em QUALQUER nível e usa o pai
    for p in Path(input_dir).rglob("Topcon_Maestro2"):
        if p.is_dir():
            return str(p.parent)
    return input_dir


def find_inference_dir(input_dir: str, training_root: str = "") -> str:
    for cand in [
        os.path.join(input_dir, "ref"),            # server: /app/input/ref
        os.path.join(input_dir, "data", "ref"),
        os.path.join(input_dir, "val", "images"),
        os.path.join(input_dir, "val"),
        os.path.join(input_dir, "test"),
        os.path.join(input_dir, "testing_data"),
        os.path.join(input_dir, "images"),
    ]:
        if os.path.isdir(cand) and any(Path(cand).rglob("*-image.png")):
            return cand
    # fallback: 1º diretório com imagens que NÃO esteja sob o training_root
    # (evita inferir nas próprias imagens de treino se o layout for inesperado)
    tr = os.path.abspath(training_root) if training_root else None
    for p in sorted(Path(input_dir).rglob("*-image.png")):
        d = os.path.abspath(str(p.parent))
        if not tr or not d.startswith(tr):
            return str(p.parent)
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
    infer_dir  = find_inference_dir(input_dir, data_root)
    ckpt_out   = os.path.join(sub_dir, "checkpoints", "unet_maestro2_semi.pth")
    pseudo_dir = os.path.join(sub_dir, "checkpoints", "pseudo")

    log(f"data_root    = {data_root}")
    log(f"infer_dir    = {infer_dir}")

    # Backup do checkpoint embarcado ANTES de treinar (o treino sobrescreve ckpt_out).
    # Serve de fallback de inferência se o treino não produzir nada dentro do tempo.
    fallback_ckpt = None
    _shipped = find_pretrained(sub_dir)
    if _shipped and os.path.exists(_shipped):
        import shutil
        fallback_ckpt = os.path.join(sub_dir, "checkpoints", "_fallback.pth")
        shutil.copy2(_shipped, fallback_ckpt)
        if os.path.exists(_shipped + ".arch.json"):
            shutil.copy2(_shipped + ".arch.json", fallback_ckpt + ".arch.json")
        log(f"fallback embarcado salvo em {fallback_ckpt}")

    def elapsed_min():
        return (time.time() - t0) / 60.0

    def remaining_min():
        return max(0, TOTAL_BUDGET_MIN - elapsed_min())

    def run_ok(cmd):  # treino de membro: falha não derruba a submissão inteira
        try:
            run(cmd, cwd=sub_dir); return True
        except subprocess.CalledProcessError as e:
            log(f"[AVISO] membro falhou ({e}) — sigo com os demais"); return False

    # ── ENSEMBLE multi-resolução (validado offline: +0.025 Mácula, +0.022 WideField vs único) ──
    # Cada membro: supervisionado + semi (widefield2), em resolução diferente (diversidade).
    # Orçamento POR MEMBRO é dinâmico (sobra / membros restantes) → robusto a overrun.
    # Configurável por env var DAOCT_ENSEMBLE = "res:canais res:canais ..." (sem edição de código):
    #   single big384 (comprovado 0.75): DAOCT_ENSEMBLE="384:48,96,192,384"
    #   ensemble + big:                  DAOCT_ENSEMBLE="384:48,96,192,384 256:16,32,64,128 512:16,32,64,128"
    _env = os.environ.get("DAOCT_ENSEMBLE", "").strip()
    if _env:
        ENSEMBLE = [(int(t.split(":")[0]), t.split(":")[1]) for t in _env.split()]
    else:
        ENSEMBLE = [(256, "16,32,64,128"), (384, "16,32,64,128"), (512, "16,32,64,128")]
    log(f"ENSEMBLE = {ENSEMBLE}")
    INFER_RESERVE_MIN = 15
    n = len(ENSEMBLE)
    trained = []
    for i, (res, channels) in enumerate(ENSEMBLE):
        slot = max(6.0, (remaining_min() - INFER_RESERVE_MIN) / max(1, n - i))
        out_i = os.path.join("checkpoints", f"m{i}")
        ckpt_i = os.path.join(sub_dir, out_i, "unet_maestro2_semi.pth")
        log(f"── membro {i+1}/{n}: {res}px {channels} | orçamento ~{slot:.1f} min ──")
        ok = run_ok([
            py(), "train_daoct.py", "--data_root", data_root, "--out", out_i,
            "--epochs", "150", "--batch_size", "16", "--img_size", str(res), "--channels", channels,
            "--amp", "auto", "--aug", "widefield2", "--workers", "0", "--seed", str(SEED + i),
            "--time_budget_min", f"{slot * 0.55:.1f}",
        ])
        if ok and os.path.exists(ckpt_i):  # semi (warm-start no supervisionado do próprio membro)
            semi_budget = max(2.0, min(slot * 0.45, remaining_min() - INFER_RESERVE_MIN))
            run_ok([
                py(), "train_daoct_semi.py", "--data_root", data_root, "--teacher", ckpt_i,
                "--out", out_i, "--pseudo_dir", os.path.join(sub_dir, out_i, "pseudo"),
                "--epochs", "100", "--batch_size", "16", "--img_size", str(res), "--channels", channels,
                "--amp", "auto", "--aug", "widefield2", "--conf_threshold", "0.82", "--semi_weight", "0.5",
                "--workers", "0", "--seed", str(SEED + i), "--time_budget_min", f"{semi_budget:.1f}",
            ])
        if os.path.exists(ckpt_i):
            trained.append(ckpt_i)
        log(f"membro {i+1}: {'ok' if os.path.exists(ckpt_i) else 'sem checkpoint'} ({elapsed_min():.1f} min decorridos)")

    # ── Inferência por ENSEMBLE (média de softmax) + TTA + refine + native ──
    models = trained if trained else ([fallback_ckpt] if fallback_ckpt else [ckpt_out])
    log(f"Ensemble de {len(models)} modelo(s) → inferência em {infer_dir} → {output_dir}")
    run([
        py(), "infer_ensemble.py",
        "--models", *models,
        "--input_dir", infer_dir,
        "--output_dir", output_dir,
        "--refine", "--native_size", "--tta",
    ], cwd=sub_dir)

    log(f"Concluído em {elapsed_min():.1f} min total.")


if __name__ == "__main__":
    main()
