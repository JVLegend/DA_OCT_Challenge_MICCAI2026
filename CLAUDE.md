# CLAUDE.md — instruções para Claude Code neste repositório

Projeto: **DA-OCT Challenge (MICCAI 2026)** — segmentação de retina OCT cross-vendor (domain adaptation).

> ## 🎯 TAREFA ATUAL (endgame — 27/06/2026)
> Estamos em **🥉 3º lugar (0.74)**. **4/5 balas usadas → resta 1 bala regular + a final.** Por isso
> estamos **SEGURANDO a última bala** e validando offline antes de gastar. **NÃO submeta nada.**
>
> **Sua tarefa (se tiver GPU): rodar o experimento de resolução e reportar.**
> 1. Garanta ambiente + dados (ver RUNBOOK_5090.md §Passo 0-2; se já rodou aqui antes, já está pronto).
> 2. `bash scripts/exp_resolution.sh` — treina 256/384/512px (16-128, widefield2) e imprime a tabela
>    do proxy. Na 5090 leva ~20-40min.
> 3. **Cole a tabela final** (e/ou atualize `STATUS.md` na seção endgame) e **commit + push** dos
>    `results/` + `STATUS.md`. Decisão (do João): se **512 > 384 > 256** (sobretudo no `plain`), a
>    bala #5 será 512px no arch grande 48-384; senão, descartamos 512 e tentamos outro lever.
> 4. **Não** monte nem suba submissão — só o experimento. O João decide a bala.

## Detecte o ambiente primeiro
Rode `nvidia-smi`:

- **Tem GPU NVIDIA (ex.: a RTX 5090)** → máquina de treino. Faça a **TAREFA ATUAL** acima. (O
  [RUNBOOK_5090.md](RUNBOOK_5090.md) descreve o pipeline completo de submissão, mas **agora não
  submetemos** — só rodamos o experimento de resolução.)

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
