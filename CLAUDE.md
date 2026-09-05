# CLAUDE.md — instruções para Claude Code neste repositório

Projeto: **DA-OCT Challenge (MICCAI 2026)** — segmentação de retina OCT cross-vendor (domain adaptation).

> ## 🚨 FINAL É SEGUNDA 07/09 — enfileirar nesse dia
> O JV competiu como time próprio e ficou em **4º lugar (0.75)** — fase de submissão dele encerrada (5/5).
>
> **Regra da final (e-mail dos organizadores):** 1 submissão única = a MELHOR da fase, **algoritmo INALTERADO
> (NÃO um modelo novo)**. → **A final do JV = re-submeter o `submission_round7_tta.zip` (0.75), sem mudar nada**
> (código na tag git `final-submission-round7-tta`). No Fact Sheet, Docker Image = `bearceb/daoct-baseline:latest`.
> ⛔ NÃO submeter o ensemble na final. Detalhe em **[STATUS.md](STATUS.md)** (topo).
>
> **Se você está na máquina do Dr. Sakuno** (time próprio, 5 balas frescas): siga o **[PLAN_SAKUNO.md](PLAN_SAKUNO.md)**
> — playbook completo (o que foi validado/descartado, como montar cada submissão, plano das 5 balas; ele PODE
> testar o ensemble nas balas dele). **Valide offline no proxy ANTES de cada bala.** A final dele = a melhor bala dele, inalterada.
>
> **Se você está na máquina do JV** (sem mais balas): só re-submeter o round7_tta na final (ver STATUS.md).

## Detecte o ambiente primeiro
Rode `nvidia-smi`:

- **Tem GPU NVIDIA (ex.: a RTX 5090 do Dr. Sakuno)** → máquina de treino/submissão. Siga o
  **[PLAN_SAKUNO.md](PLAN_SAKUNO.md)** (campanha de 5 submissões) + **[RUNBOOK_5090.md](RUNBOOK_5090.md)** (setup).

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
