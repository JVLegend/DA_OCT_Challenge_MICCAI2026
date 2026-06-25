#!/bin/bash
# =============================================================================
# Pipeline AUTÔNOMO da RTX 5090 — roda tudo ponta-a-ponta e salva resultados no Git.
# Pensado pra ser disparado por um Claude Code que leu o RUNBOOK_5090.md.
#
#   1. ambiente (venv + torch cu128 + deps)
#   2. dados (baixa do Google Drive se faltar) -> data/starter_kit
#   3. checa GPU (Blackwell / bf16)
#   4. ROUND 1: treino forte arch baseline (drop-in) + avalia
#   5. ROUND 2: treino arch maior + alta resolução + avalia
#   6. grava results/ + STATUS.md + log de experimentos
#   7. commit + push (se o git estiver autenticado)
#
# Uso: bash scripts/run_all_5090.sh
# Dado: defina DAOCT_DATA_URL=<link Drive> OU crie data/DATA_URL.txt com o link.
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
VENV="$ROOT/.venv/bin/python"
log(){ echo -e "\n\033[1;36m== $* ==\033[0m"; }

# ---------- 1. ambiente ----------
log "1/7 Ambiente"
command -v uv >/dev/null || pip install -q uv
[ -d .venv ] || uv venv --python 3.11 .venv
# torch com CUDA 12.8 (Blackwell/sm_120) — só instala se ainda não enxerga a GPU
if ! "$VENV" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  uv pip install --python "$VENV" torch --index-url https://download.pytorch.org/whl/cu128
fi
uv pip install --python "$VENV" -r requirements.txt "gdown>=5" >/dev/null

# ---------- 2. dados ----------
log "2/7 Dados"
if [ ! -d "data/starter_kit/app_ingestion/input_data/train" ]; then
  URL="${DAOCT_DATA_URL:-}"; [ -z "$URL" ] && [ -f data/DATA_URL.txt ] && URL="$(tr -d '[:space:]' < data/DATA_URL.txt)"
  if [ -z "$URL" ]; then
    echo "[ERRO] Dados ausentes e sem link. Defina DAOCT_DATA_URL ou crie data/DATA_URL.txt com o link do Google Drive."
    echo "       (Peça o link ao João, salve em data/DATA_URL.txt e rode de novo.)"; exit 2
  fi
  mkdir -p data
  # extrai o file-id de links tipo /file/d/<id>/view (robusto em qualquer versão do gdown)
  FID="$(echo "$URL" | sed -n 's#.*/file/d/\([^/]*\)/.*#\1#p')"
  [ -z "$FID" ] && FID="$(echo "$URL" | sed -n 's#.*[?&]id=\([^&]*\).*#\1#p')"
  if [ -n "$FID" ]; then DL="https://drive.google.com/uc?id=$FID"; else DL="$URL"; fi
  echo "[INFO] baixando dados do Drive ($DL)..."
  "$VENV" -m gdown "$DL" -O data/starter_kit.zip
  mkdir -p data/starter_kit && unzip -q -o data/starter_kit.zip -d data/starter_kit
fi
TRAIN="data/starter_kit/app_ingestion/input_data/train"
[ -d "$TRAIN/Topcon_Maestro2" ] || { echo "[ERRO] estrutura de dados inesperada em $TRAIN"; exit 2; }
echo "[OK] dados prontos."

# ---------- 3. GPU ----------
log "3/7 GPU"
"$VENV" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA indisponível — confira driver/cu128"
print("gpu:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
PY

run_round(){ # $1=nome $2=args extras
  local name="$1"; shift
  log "Treino $name"
  "$VENV" scripts/train_daoct.py --data_root "$TRAIN" --out "models/$name" \
    --amp auto --aug strong --workers 8 "$@"
  log "Avaliação $name"
  local score; score="$(bash scripts/eval_local.sh "models/$name/unet_maestro2_semi.pth" | tail -1)"
  echo "$score" > "models/$name/score.json"
  mkdir -p "results/$name"
  cp -f "models/$name/train_report.json" "results/$name/" 2>/dev/null || true
  cp -f "models/$name/score.json" "results/$name/" 2>/dev/null || true
  # checkpoint pequeno também vai pro git (pra você baixar o modelo)
  local sz; sz=$(du -m "models/$name/unet_maestro2_semi.pth" | cut -f1)
  [ "$sz" -lt 80 ] && cp -f "models/$name/unet_maestro2_semi.pth" "results/$name/" || echo "[INFO] checkpoint $name ${sz}MB > 80MB, não versionado"
  echo "$score"
}

# ---------- 4 & 5. rounds ----------
S1="$(run_round round1 --epochs 200 --batch_size 24 --img_size 256)"      # arch baseline (drop-in)
S2="$(run_round round2 --epochs 250 --batch_size 16 --img_size 384 --channels 32,64,128,256)"  # maior + alta res

# ---------- 6. status + log ----------
log "6/7 Status & log"
"$VENV" - "$S1" "$S2" <<'PY'
import json, sys, datetime
s1=json.loads(sys.argv[1]); s2=json.loads(sys.argv[2])
ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
m1=s1.get("macula_score",0); m2=s2.get("macula_score",0)
best="round2" if m2>=m1 else "round1"
open("STATUS.md","w").write(f"""# STATUS — execução autônoma na 5090

Última atualização: {ts}

## Resultados (val local = Maestro2/Mácula/saudável; proxy, não leaderboard)
| Round | arch / img | macula_score | final(local) |
|---|---|---|---|
| round1 | baseline 16-128 / 256 | {m1:.4f} | {s1.get('final_score',0):.4f} |
| round2 | 32-256 / 384 | {m2:.4f} | {s2.get('final_score',0):.4f} |

**Melhor até agora: {best}** (baseline de referência era 0.2234).
Checkpoints e relatórios em `results/round1/` e `results/round2/`.

## Próximos passos sugeridos (para o João revisar)
1. Se round2 > round1, levar a arch maior + 384px adiante; senão investigar overfit/VRAM.
2. **F2 — domain adaptation real:** implementar semi-supervisão (pseudo-labels em
   Spectralis/Cirrus/Maestro2_unlabeled) — é onde está o ganho de generalização (Triton/WideField).
3. Construir val cross-vendor (segurar 1 vendor como proxy de "unseen") — o val local só mede Maestro2.
4. Endurecer o pipeline de submissão (treino in-container < 2h, zip com arquivos na raiz). Ver docs/07.

## Como continuar automaticamente
O Claude Code nesta máquina pode rodar a próxima rodada (ex.: semi-supervisão) e dar push de novo.
""")
# log de experimentos (append)
line1=f"| {ts} | 5090 round1 (200ep, 256, baseline) | {m1:.4f} | {s1.get('final_score',0):.4f} | autônomo |\n"
line2=f"| {ts} | 5090 round2 (250ep, 384, 32-256)   | {m2:.4f} | {s2.get('final_score',0):.4f} | autônomo |\n"
open("docs/06_log_experimentos.md","a").write(line1+line2)
print("STATUS.md e log atualizados.")
PY

# ---------- 7. commit/push ----------
log "7/7 Commit & push"
git add results/ STATUS.md docs/06_log_experimentos.md scripts/ 2>/dev/null || true
git commit -q -m "5090 autônomo: round1+round2 (results, status, log)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>" || echo "[INFO] nada novo p/ commit"
if git push -q 2>/dev/null; then echo "[OK] push feito."; else
  echo "[AVISO] push falhou (git sem auth?). Rode 'gh auth login' ou configure um token e 'git push'."; fi

log "FIM — veja STATUS.md"
