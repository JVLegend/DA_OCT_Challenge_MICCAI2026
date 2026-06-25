# Prompt de handoff — Claude Code na máquina da RTX 5090

> Copie TUDO abaixo da linha para um Claude Code rodando **na máquina com a RTX 5090**,
> a partir da raiz do repositório `DA_OCT_Challenge_MICCAI2026`.

---

Você é um Claude Code rodando numa máquina Linux/Windows com uma **RTX 5090 (Blackwell, sm_120)**.
Está na raiz do repositório `DA_OCT_Challenge_MICCAI2026`. Sua missão é o **treino pesado (F1)** do
modelo de segmentação de retina OCT do DA-OCT Challenge (MICCAI 2026) e devolver um checkpoint
melhor que o baseline. Trabalhe com critério de engenharia (pense antes de codar, mudanças
cirúrgicas, simplicidade, e itere até bater o critério de sucesso).

## Contexto da competição
- Segmentação semântica **10 classes** (8 camadas da retina + 2 fundos), valores de máscara 0–9.
- Treino rotulado **só** no Topcon Maestro2; Spectralis/Cirrus/Maestro2_unlabeled vêm SEM label.
- Placar real (em `data/starter_kit/app_scoring/program/metrics.py`): doente pesa 0.7, metade do
  score é WideField, penaliza não generalizar pro vendor oculto **Triton**, MASD severo (τ=0.02).
- O baseline **não faz domain adaptation** (5 épocas, sem augmentation) → muito teto.
- Leia `docs/05_analise_baseline_e_estrategia.md` e `docs/06_log_experimentos.md` antes de começar.
- **Número-base a bater:** `macula_score = 0.2234` no val local (script abaixo).

## Pré-requisitos (verifique primeiro; pare com mensagem clara se faltar)
1. Os dados precisam estar em `data/starter_kit/` (descompactados). Se houver só o zip
   `miccai_satelite_starting_kit_*.zip` em `data/`, descompacte:
   `mkdir -p data/starter_kit && unzip -q -o data/<zip> -d data/starter_kit`
   Confirme que existe `data/starter_kit/app_ingestion/input_data/train/Topcon_Maestro2/`.
2. Se nem o zip estiver lá, PARE e peça o arquivo ao João (não há como baixar o dado protegido aqui).

## Passo 1 — Ambiente Blackwell (CUDA 12.8)
```bash
# venv 3.11 com uv (instale uv se faltar: pip install uv)
uv venv --python 3.11 .venv
# TORCH com CUDA 12.8 — OBRIGATÓRIO pra sm_120 (5090). Sem isso o torch não acha a GPU.
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.txt
```
Valide a GPU (precisa imprimir o nome da 5090, bf16=True e um matmul em cuda sem erro):
```bash
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA indisponível — confira driver/cu128"
print("gpu:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
x = torch.randn(2048,2048, device="cuda"); print("matmul ok:", (x@x).sum().item() is not None)
PY
```

## Passo 2 — Sanity na GPU (rápido)
```bash
.venv/bin/python scripts/train_daoct.py \
  --data_root data/starter_kit/app_ingestion/input_data/train \
  --out models --epochs 2 --limit 32 --batch_size 8 --amp auto
```
Confirme no cabeçalho: `device: cuda (... 5090)` e `AMP enabled=True dtype=torch.bfloat16`.

## Passo 3 — Treino pesado F1 (o trabalho de verdade)
O `scripts/train_daoct.py` já é portável e seleciona device/precisão sozinho. Rode forte:
```bash
.venv/bin/python scripts/train_daoct.py \
  --data_root data/starter_kit/app_ingestion/input_data/train \
  --out models --epochs 200 --batch_size 24 --img_size 256 \
  --aug strong --amp auto --workers 8 --lr 1e-3
```
- Mantenha a **arquitetura padrão** (UNet 16/32/64/128) para o checkpoint ser **drop-in** no
  infer do kit. Só mude `--channels` se também atualizar `build_model` em
  `data/starter_kit/app_ingestion/program/infer_test_monai.py` (e documentar).
- Se aparecer **NaN** na loss (risco de mixed precision), tente `--amp off` ou `--amp fp16`.
- Ajuste `--batch_size` ao limite de VRAM (a 5090 tem 32 GB; pode subir bem).

## Passo 4 — Avaliar e registrar
```bash
# usa nosso checkpoint no infer+scoring oficiais do kit (val local = Maestro2/Mácula/saudável)
bash scripts/run_baseline_local.sh models/unet_maestro2_semi.pth
```
- Anote o `macula_score` em `docs/06_log_experimentos.md` (nova linha na tabela, com a data e a config).
- Guarde `models/unet_maestro2_semi.pth` e `models/train_report.json`.

## Critério de sucesso
- `macula_score` no val local **claramente acima de 0.2234** (idealmente > 0.6).
- `models/unet_maestro2_semi.pth` salvo + `train_report.json` + linha no log de experimentos.

## Devolver pro João
- O checkpoint é pequeno (~poucos MB). Devolva por: `git push` (se houver remote) OU copie
  `models/unet_maestro2_semi.pth` + `models/train_report.json` de volta pro Mac mini.
- **NÃO** versione `data/` nem `models/` no git (já estão no `.gitignore`). **NÃO** toque no
  vault Obsidian (outra máquina). Faça commit só de código/docs, e só se houver remote configurado.

## Stretch (se sobrar GPU/tempo) — F2
- Augmentation de aparência já está ligada (gamma/contraste/ruído) = robustez cross-vendor.
- Próximo ganho real é **semi-supervisão com os não rotulados** (pseudo-labels em
  Spectralis/Cirrus/Maestro2_unlabeled) + montar um val cross-vendor. A flag `--semi` é stub:
  se for implementar, faça num branch e relate o que funcionou pro João decidir.

Ao terminar, escreva um resumo curto: melhor `val_dice`, `macula_score`, config vencedora e onde
ficou o checkpoint.
