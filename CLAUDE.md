# CLAUDE.md — instruções para Claude Code neste repositório

Projeto: **DA-OCT Challenge (MICCAI 2026)** — segmentação de retina OCT cross-vendor (domain adaptation).

## Detecte o ambiente primeiro
Rode `nvidia-smi`:

- **Tem GPU NVIDIA (ex.: a RTX 5090)** → você está na máquina de treino pesado. **Execute o
  [RUNBOOK_5090.md](RUNBOOK_5090.md) de forma autônoma**: prepare o ambiente, baixe os dados do
  Google Drive (link em `data/DATA_URL.txt` ou `$DAOCT_DATA_URL`; se faltar, peça ao João), rode
  `bash scripts/run_all_5090.sh`, revise os resultados, atualize `STATUS.md`, e se a GPU estiver
  livre **adiante a próxima rodada** (F2 / semi-supervisão). Faça **commit + push** dos resultados.

- **Sem GPU (Mac/dev)** → é máquina de desenvolvimento. Veja o [README.md](README.md) e os `docs/`.
  Dá pra treinar a F1 no MPS (`scripts/train_daoct.py`), mas o pesado vai pra 5090/Kaggle.

## Regras
- **Nunca** versione `data/` nem dados/checkpoints grandes — eles são protegidos/pesados e o
  `.gitignore` já exclui. Resultados (relatórios, scores, status, checkpoints pequenos) vão em `results/`.
- Mantenha a arquitetura do round1 padrão (drop-in no infer do kit do desafio).
- A regra do desafio é open-source BSD-2 (já há `LICENSE`). Repo é público.
- Antes de gastar uma submissão real (só **5** na fase atual): valide localmente. Ver `docs/07_insights_forum.md`.

## Mapa rápido
- `scripts/run_all_5090.sh` — pipeline autônomo completo (env → dados → treino → eval → results → push).
- `scripts/train_daoct.py` — treino portável (Mac/T4/5090, device+AMP automáticos).
- `scripts/infer_daoct.py` — inferência arch-aware + descoberta robusta de imagens.
- `scripts/eval_local.sh` — avalia um checkpoint no val local (proxy).
- `RUNBOOK_5090.md` — o plano detalhado pra esta máquina.
- `STATUS.md` — placar atual e próximos passos.
