#!/bin/bash
# Roda o loop completo localmente (CPU): inferência -> scoring -> imprime score.
# Uso: scripts/run_baseline_local.sh [CHECKPOINT.pth]
# Sem argumento, usa o checkpoint baseline do starter kit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv/bin/python"
KIT="$ROOT/data/starter_kit"
CKPT="${1:-$KIT/app_ingestion/program/checkpoints/unet_maestro2_semi.pth}"

VAL_IMAGES="$KIT/app_ingestion/input_data/val/images"
RES_DIR="$KIT/app_scoring/input/res"
OUT_DIR="$KIT/app_scoring/output"

echo "[1/2] Inferência com checkpoint: $CKPT"
rm -f "$RES_DIR"/*-mask.png 2>/dev/null || true
mkdir -p "$RES_DIR"
"$VENV" "$KIT/app_ingestion/program/infer_test_monai.py" \
  --input_dir "$VAL_IMAGES" --output_dir "$RES_DIR" --model_path "$CKPT"

echo "[2/2] Scoring local (173 Maestro2/Mácula/saudáveis)"
( cd "$KIT/app_scoring/program" && "$VENV" scoring.py "$KIT/app_scoring/input" "$OUT_DIR" >/dev/null 2>&1 )
echo "Resultado:"
cat "$OUT_DIR/scores.json"; echo

# NOTA: o val local só cobre Maestro2/Mácula/saudável (WideField=0).
# Não é a nota do leaderboard — serve pra comparar experimentos entre si.
