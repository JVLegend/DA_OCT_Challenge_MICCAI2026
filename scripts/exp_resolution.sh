#!/bin/bash
# Sweep de RESOLUÇÃO (256 / 384 / 512), arch 16-128 + widefield2, medido no proxy.
# Hipótese (lacuna p/ o 2º): mais resolução -> bordas mais precisas -> melhor MASD (metade da nota),
# em Mácula E WideField. Se 512 > 384 > 256 no proxy, a bala #5 = 512px (no big arch 48-384).
#
# 512px é PESADO: ideal rodar na 5090 (--device auto pega cuda) ou Kaggle. No Mac MPS leva horas.
# Auto-suficiente: treina 256/384/512 do zero (16-128/widefield2). Skip se já existir.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
VENV="$ROOT/.venv/bin/python"
DR="data/starter_kit/app_ingestion/input_data/train"

for RES in 256 384 512; do
  D="models/exp_res$RES"
  if [ -f "$D/train_report.json" ]; then echo "## SKIP res=$RES (já treinado)"; continue; fi
  echo "## TREINO img_size=$RES (16-128, widefield2)"
  "$VENV" scripts/train_daoct.py --data_root "$DR" --out "$D" \
    --epochs 100 --batch_size 8 --img_size "$RES" --channels 16,32,64,128 \
    --aug widefield2 --workers 0 --device auto --amp auto --val_frac 0.15 --seed 42 2>&1 | tail -2
done

echo ""
echo "############## RESULTADO (sweep de resolução, aug=widefield2) ##############"
printf "%-10s | %-8s | %-10s | %-10s\n" "img_size" "plain" "cosine.30" "radial.50"
for PAIR in "256:models/exp_res256" "384:models/exp_res384" "512:models/exp_res512"; do
  RES="${PAIR%%:*}"; CK="${PAIR##*:}/unet_maestro2_semi.pth"
  [ -f "$CK" ] || { printf "%-10s | (faltando)\n" "$RES"; continue; }
  OC=$("$VENV" scripts/eval_proxy_widefield.py --model_path "$CK" --warp cosine --amp 0.30 2>/dev/null)
  OR=$("$VENV" scripts/eval_proxy_widefield.py --model_path "$CK" --warp radial --amp 0.50 2>/dev/null)
  PL=$(echo "$OC" | grep "SEM warp" | grep -oE "[0-9.]+$")
  C=$(echo "$OC" | grep "COM warp" | grep -oE "[0-9.]+$")
  R=$(echo "$OR" | grep "COM warp" | grep -oE "[0-9.]+$")
  printf "%-10s | %-8s | %-10s | %-10s\n" "$RES" "$PL" "$C" "$R"
done
echo "############## FIM ##############"
echo "Se 512 ganha (sobretudo no 'plain', que reflete a Mácula/MASD) -> bala #5 = 512px no big arch."
