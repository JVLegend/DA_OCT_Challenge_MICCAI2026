#!/bin/bash
# Teste de capacidade: arch pequena (16-128, já treinada como exp_widefield2) vs grande (48-384),
# ambas aug=widefield2 / 256px. Se a grande ganha já no proxy pequeno, ganha mais no servidor
# (que tem ~20x mais dados). Caveat: 230 imgs no Mac subestima o ganho de capacidade.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
VENV="$ROOT/.venv/bin/python"
DR="data/starter_kit/app_ingestion/input_data/train"

# arch grande (a pequena 16-128 já existe em models/exp_widefield2)
if [ ! -f "models/exp_big384/train_report.json" ]; then
  echo "## TREINO arch=48,96,192,384 (widefield2, 256px)"
  "$VENV" scripts/train_daoct.py --data_root "$DR" --out "models/exp_big384" \
    --epochs 100 --batch_size 6 --img_size 256 --channels 48,96,192,384 \
    --aug widefield2 --workers 0 --device mps --amp off --val_frac 0.15 --seed 42 2>&1 | tail -2
else
  echo "## SKIP big384 (já treinado)"
fi

echo ""
echo "############## RESULTADO (capacidade, aug=widefield2) ##############"
printf "%-22s | %-8s | %-10s | %-10s\n" "modelo" "plain" "cosine.30" "radial.50"
declare -A MODELS=( ["16-128 (small)"]="models/exp_widefield2/unet_maestro2_semi.pth" ["48-384 (big)"]="models/exp_big384/unet_maestro2_semi.pth" )
for NAME in "16-128 (small)" "48-384 (big)"; do
  CK="${MODELS[$NAME]}"
  [ -f "$CK" ] || { printf "%-22s | (faltando)\n" "$NAME"; continue; }
  OC=$("$VENV" scripts/eval_proxy_widefield.py --model_path "$CK" --warp cosine --amp 0.30 2>/dev/null)
  OR=$("$VENV" scripts/eval_proxy_widefield.py --model_path "$CK" --warp radial --amp 0.50 2>/dev/null)
  PL=$(echo "$OC" | grep "SEM warp" | grep -oE "[0-9.]+$")
  C=$(echo "$OC" | grep "COM warp" | grep -oE "[0-9.]+$")
  R=$(echo "$OR" | grep "COM warp" | grep -oE "[0-9.]+$")
  printf "%-22s | %-8s | %-10s | %-10s\n" "$NAME" "$PL" "$C" "$R"
done
echo "############## FIM ##############"
echo "Se big >= small nos 3 -> capacidade ajuda mesmo com poucos dados; no servidor (mais dados) ajuda mais."
