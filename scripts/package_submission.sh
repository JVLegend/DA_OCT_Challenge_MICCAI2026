#!/bin/bash
# Empacota a pasta submission/ num ZIP com arquivos na RAIZ.
# Regra do desafio (docs/07 thread #118): entrypoint.sh deve estar na raiz do zip,
# não dentro de uma subpasta (Finder/Compress do macOS embrulha numa pasta extra → falha).
#
# Uso: bash scripts/package_submission.sh [versao]
# Ex.: bash scripts/package_submission.sh round3_semi
#
# Valide o zip antes de submeter:
#   unzip -l submission_round3_semi.zip | head -20
#   # deve mostrar entrypoint.sh, main.py, etc. na raiz — sem prefixo de pasta.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VER="${1:-submission}"
OUT="$ROOT/submission_${VER}.zip"

# Atualiza os scripts no submission/ com as versões mais recentes de scripts/
cp "$ROOT/scripts/train_daoct.py"      "$ROOT/submission/"
cp "$ROOT/scripts/train_daoct_semi.py" "$ROOT/submission/"
cp "$ROOT/scripts/infer_daoct.py"      "$ROOT/submission/"

# Atualiza checkpoint com o melhor disponível
CKPT_SRC=""
for candidate in \
    "$ROOT/models/round3_semi/unet_maestro2_semi.pth" \
    "$ROOT/models/round2/unet_maestro2_semi.pth" \
    "$ROOT/models/round1/unet_maestro2_semi.pth"; do
    if [ -f "$candidate" ]; then
        CKPT_SRC="$candidate"; break
    fi
done

if [ -n "$CKPT_SRC" ]; then
    cp "$CKPT_SRC" "$ROOT/submission/checkpoints/unet_maestro2_semi.pth"
    [ -f "${CKPT_SRC}.arch.json" ] && \
        cp "${CKPT_SRC}.arch.json" "$ROOT/submission/checkpoints/unet_maestro2_semi.pth.arch.json"
    echo "[OK] checkpoint: $CKPT_SRC"
fi

# Empacota: cd submission/ e zipa o CONTEÚDO (.) para que arquivos fiquem na raiz
cd "$ROOT/submission"
zip -r "$OUT" . -x "*.pyc" -x "__pycache__/*" -x ".DS_Store"

echo ""
echo "[OK] ZIP gerado: $OUT"
echo ""
echo "Verificação (entrypoint.sh deve aparecer na raiz, sem prefixo de pasta):"
unzip -l "$OUT" | grep -E "entrypoint|main\.py|infer_daoct|unet_maestro2_semi\.pth$" | head -20
