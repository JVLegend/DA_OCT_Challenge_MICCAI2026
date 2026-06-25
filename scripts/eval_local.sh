#!/bin/bash
# Avalia um checkpoint (qualquer arch) no val local via infer_daoct + scoring oficial do kit.
# Uso: scripts/eval_local.sh <CHECKPOINT.pth>
# Imprime o JSON de score. (Val local = 173 Maestro2/Mácula/saudável; é proxy, não o leaderboard.)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv/bin/python"
KIT="$ROOT/data/starter_kit"
CKPT="${1:?uso: eval_local.sh <checkpoint.pth>}"

RES="$KIT/app_scoring/input/res"
OUT="$KIT/app_scoring/output"
rm -f "$RES"/*-mask.png 2>/dev/null || true; mkdir -p "$RES"

"$VENV" "$ROOT/scripts/infer_daoct.py" \
  --input_dir "$KIT/app_ingestion/input_data/val/images" --output_dir "$RES" --model_path "$CKPT"
( cd "$KIT/app_scoring/program" && "$VENV" scoring.py "$KIT/app_scoring/input" "$OUT" >/dev/null 2>&1 )
cat "$OUT/scores.json"; echo
