#!/bin/bash
# Experimento v2: strong vs widefield vs widefield2, medido em DOIS warps (anti-overfit).
#   cosine 0.30  : curva vertical global
#   radial 0.50  : barrel 2D (geometria bem diferente)
# Sucesso = widefield2 melhora nos DOIS warps vs strong, com 'plain' (Mácula) intacto.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
VENV="$ROOT/.venv/bin/python"
DR="data/starter_kit/app_ingestion/input_data/train"

for AUG in strong widefield widefield2; do
  if [ -f "models/exp_$AUG/train_report.json" ]; then
    echo "## SKIP $AUG (val_dice=$("$VENV" -c "import json;print(round(json.load(open('models/exp_$AUG/train_report.json'))['best_val_dice'],4))"))"
    continue
  fi
  echo "## TREINO aug=$AUG"
  "$VENV" scripts/train_daoct.py --data_root "$DR" --out "models/exp_$AUG" \
    --epochs 100 --batch_size 8 --img_size 256 --channels 16,32,64,128 \
    --aug "$AUG" --workers 0 --device mps --amp off --val_frac 0.15 --seed 42 2>&1 | tail -2
done

echo ""
echo "############## RESULTADO ##############"
printf "%-12s | %-8s | %-12s | %-12s\n" "aug" "plain" "cosine.30" "radial.50"
for AUG in strong widefield widefield2; do
  CK="models/exp_$AUG/unet_maestro2_semi.pth"
  OC=$("$VENV" scripts/eval_proxy_widefield.py --model_path "$CK" --warp cosine --amp 0.30 2>/dev/null)
  OR=$("$VENV" scripts/eval_proxy_widefield.py --model_path "$CK" --warp radial --amp 0.50 2>/dev/null)
  PL=$(echo "$OC" | grep "SEM warp" | grep -oE "[0-9.]+$")
  C=$(echo "$OC" | grep "COM warp" | grep -oE "[0-9.]+$")
  R=$(echo "$OR" | grep "COM warp" | grep -oE "[0-9.]+$")
  printf "%-12s | %-8s | %-12s | %-12s\n" "$AUG" "$PL" "$C" "$R"
done
echo "############## FIM ##############"
echo "Escolher a aug com maior cosine.30 E radial.50, mantendo plain ~igual ao strong."
