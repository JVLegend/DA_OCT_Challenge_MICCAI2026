#!/bin/bash
# Sweep de boundary loss (HausdorffDTLoss): pesos 0.01 e 0.02 vs baseline (sem boundary),
# todos widefield2 / 16-128 / 256px. O HD otimiza a fronteira das camadas = MASD, nossa maior lacuna
# (Mácula). Mede no proxy (plain reflete MASD/Mácula).
#   HD/Dice ~53x -> peso 0.01 = boundary ~0.5x do Dice; 0.02 = ~1x.
#   warmup 0.3 (boundary só após 30% das épocas, deixa o Dice estabilizar).
# OBS: no MPS o termo HD cai no CPU (float64); no servidor/5090 (CUDA) roda nativo (mais rápido).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
VENV="$ROOT/.venv/bin/python"
DR="data/starter_kit/app_ingestion/input_data/train"

for W in 0.01 0.02; do
  D="models/exp_bd${W}"
  if [ -f "$D/train_report.json" ]; then echo "## SKIP bd=$W (já treinado)"; continue; fi
  echo "## TREINO boundary_weight=$W (widefield2, 16-128, 256px)"
  "$VENV" scripts/train_daoct.py --data_root "$DR" --out "$D" \
    --epochs 100 --batch_size 8 --img_size 256 --channels 16,32,64,128 \
    --aug widefield2 --boundary_weight "$W" --boundary_warmup_frac 0.3 \
    --workers 0 --device auto --amp off --val_frac 0.15 --seed 42 2>&1 | tail -2
done

echo ""
echo "############## RESULTADO (boundary loss, proxy) ##############"
printf "%-16s | %-8s | %-10s | %-10s\n" "config" "plain" "cosine.30" "radial.50"
for PAIR in "baseline:models/exp_widefield2" "bd=0.01:models/exp_bd0.01" "bd=0.02:models/exp_bd0.02"; do
  NAME="${PAIR%%:*}"; CK="${PAIR##*:}/unet_maestro2_semi.pth"
  [ -f "$CK" ] || { printf "%-16s | (faltando)\n" "$NAME"; continue; }
  OC=$("$VENV" scripts/eval_proxy_widefield.py --model_path "$CK" --warp cosine --amp 0.30 2>/dev/null)
  OR=$("$VENV" scripts/eval_proxy_widefield.py --model_path "$CK" --warp radial --amp 0.50 2>/dev/null)
  PL=$(echo "$OC" | grep "SEM warp" | grep -oE "[0-9.]+$")
  C=$(echo "$OC" | grep "COM warp" | grep -oE "[0-9.]+$")
  R=$(echo "$OR" | grep "COM warp" | grep -oE "[0-9.]+$")
  printf "%-16s | %-8s | %-10s | %-10s\n" "$NAME" "$PL" "$C" "$R"
done
echo "############## FIM ##############"
echo "Se algum bd melhora o 'plain' (=Mácula/MASD) sem regredir -> boundary loss entra na final."
