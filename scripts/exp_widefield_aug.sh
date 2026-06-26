#!/bin/bash
# Experimento OFFLINE: aug 'strong' vs 'widefield' medido no proxy de curvatura.
# Hipótese: aug geométrica forte (grid distortion + affine grande) recupera a robustez
# do modelo à curvatura (proxy do wide-field), sem regredir a Mácula.
# Modelo pequeno/rápido (16-128 / 256) só p/ medir o EFEITO da aug; o submission usa 32-256/384.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
VENV="$ROOT/.venv/bin/python"
DR="data/starter_kit/app_ingestion/input_data/train"

for AUG in strong widefield; do
  echo "############## TREINO aug=$AUG ##############"
  "$VENV" scripts/train_daoct.py --data_root "$DR" --out "models/exp_$AUG" \
    --epochs 100 --batch_size 8 --img_size 256 --channels 16,32,64,128 \
    --aug "$AUG" --workers 0 --device mps --amp off --val_frac 0.15 --seed 42 2>&1 | tail -2
done

echo ""
echo "############## RESULTADO (proxy curvatura amp=0.30) ##############"
printf "%-12s | %-10s | %-12s | %s\n" "aug" "plain" "curvatura.30" "queda"
for AUG in strong widefield; do
  OUT=$("$VENV" scripts/eval_proxy_widefield.py --model_path "models/exp_$AUG/unet_maestro2_semi.pth" --amp 0.30 2>/dev/null)
  P=$(echo "$OUT" | grep "SEM warp" | grep -oE "[0-9.]+$")
  W=$(echo "$OUT" | grep "COM curvatura" | grep -oE "[0-9.]+$")
  Q=$(echo "$OUT" | grep "QUEDA" | grep -oE "\([0-9.]+%\)")
  printf "%-12s | %-10s | %-12s | %s\n" "$AUG" "$P" "$W" "$Q"
done
echo "############## FIM ##############"
echo "Leitura: 'plain' parecido nos dois (Mácula protegida) E 'curvatura.30' do widefield > strong = aug ajuda."
