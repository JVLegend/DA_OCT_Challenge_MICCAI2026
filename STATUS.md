# STATUS — DA-OCT Challenge

Última atualização: 2026-06-26 (round1 + round2 executados na RTX 5090)

## Estado
- ✅ F0: pipeline local fechado. Baseline: **macula_score = 0.2234** (val local, proxy).
- ✅ F1: round1 + round2 treinados na RTX 5090. **Round2 é o melhor até agora.**
- ⏳ F2: domain adaptation real (semi-supervisão por pseudo-labels) — próximo passo.

## Resultados (val local = Maestro2/Mácula/saudável; proxy, não leaderboard)

| Round | arch / img | best_val_dice | macula_score | final(local) | vs baseline |
|---|---|---|---|---|---|
| baseline (F0) | UNet 5ep / 256 | — | 0.2234 | 0.1117 | referência |
| **round1 (F1)** | 16-128 / 256 | 0.8853 | 0.2337 | 0.1169 | +4.6% |
| **round2 (F1)** | 32-256 / 384 | — | **0.2394** | **0.1197** | **+7.2%** |

**Melhor até agora: round2** (macula_score 0.2394 vs baseline 0.2234).
Checkpoints em `models/round1/` e `models/round2/`. Scores em `results/round1/` e `results/round2/`.

> widefield_score = 0.0 em ambos — esperado, não temos labels cross-vendor no val local.
> O ganho real no leaderboard virá da F2 (generalização p/ Spectralis/Cirrus).

## Próximos passos
1. **F2 — domain adaptation real:** semi-supervisão por pseudo-labels (Spectralis, Cirrus, Maestro2_unlabeled). É aqui que está o ganho de generalização e o widefield_score.
2. Val cross-vendor: separar 1 vendor como proxy de "unseen" para medir generalização localmente.
3. Endurecer pipeline de submissão (treino in-container < 2h, zip com arquivos na raiz). Ver docs/07.
