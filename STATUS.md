# STATUS — DA-OCT Challenge

Última atualização: 2026-06-25 (setup; antes da 1ª execução na 5090)

## Estado
- ✅ F0: pipeline local fechado. Baseline: **macula_score = 0.2234** (val local, proxy).
- ⏳ Aguardando 1ª execução autônoma na **RTX 5090** (`scripts/run_all_5090.sh`).

## Resultados
_(serão preenchidos automaticamente pela execução na 5090 — round1/round2)_

| Round | arch / img | macula_score | final(local) |
|---|---|---|---|
| — | — | — | — |

## Próximos passos
1. Rodar `run_all_5090.sh` na 5090 (round1 baseline + round2 maior/alta-res).
2. F2 — domain adaptation real (semi-supervisão por pseudo-labels) = onde está o ganho.
3. Val cross-vendor + endurecer pipeline de submissão (treino in-container < 2h).
