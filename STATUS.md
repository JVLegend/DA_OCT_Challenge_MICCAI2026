# STATUS — DA-OCT Challenge

Última atualização: 2026-06-26 (F2 semi-supervisão concluída na RTX 5090)

## Estado
- ✅ F0: pipeline local fechado. Baseline: **macula_score = 0.2234** (val local, proxy).
- ✅ F1: round1 + round2 treinados na RTX 5090.
- ✅ F2: semi-supervisão por pseudo-labels concluída. **Round3 é o melhor até agora.**
- ⏳ F3/F4: val cross-vendor + endurecer pipeline de submissão.

## Resultados (val local = Maestro2/Mácula/saudável; proxy, não leaderboard)

| Round | arch / img | best_val_dice | macula_score | final(local) | vs baseline |
|---|---|---|---|---|---|
| baseline (F0) | UNet 5ep / 256 | — | 0.2234 | 0.1117 | referência |
| round1 (F1) | 16-128 / 256 | 0.8853 | 0.2337 | 0.1169 | +4.6% |
| round2 (F1) | 32-256 / 384 | 0.8853 | 0.2394 | 0.1197 | +7.2% |
| **round3_semi (F2)** | 32-256 / 384 + pseudo | **0.9019** | **0.2412** | **0.1206** | **+8.0%** |

**Melhor até agora: round3_semi** — pseudo-labels em 1095/1099 imagens não rotuladas (99.6%), warm-start em round2, 150 épocas, 4.8 min na 5090.

> widefield_score = 0.0 em todos — esperado, val local é 100% Maestro2. O ganho de
> generalização cross-vendor (Spectralis/Cirrus/Triton) só aparece no leaderboard real.

## Próximos passos
1. **Val cross-vendor interno:** separar 1 vendor como proxy de "unseen" para medir generalização real antes de submeter.
2. **F3 — MASD/borda:** suavização monotônica das camadas (τ=0.02 penaliza bordas imprecisas) e inferência em resolução nativa.
3. **F4 — Submissão:** endurecer descoberta de imagens (recursiva, nunca 0), zip com arquivos na raiz, treino in-container < 2h. Ver docs/07.
4. Gastar 1ª bala real só após validar localmente de forma exaustiva.
