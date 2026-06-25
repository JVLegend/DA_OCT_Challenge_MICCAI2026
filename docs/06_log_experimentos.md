# 06 — Log de Experimentos

Comparação entre runs no **val local** (`scripts/run_baseline_local.sh`).
⚠️ Val local = 173 imagens **Maestro2 / Mácula / saudável**. WideField ausente (conta 0).
Não é a nota do leaderboard — serve pra comparar experimentos entre si. A coluna que importa
pra comparar é **macula_score** (desempenho no Maestro2 fácil).

| Data | Experimento | macula_score | final (local) | Notas |
|---|---|---|---|---|
| 2026-06-25 | **F0 — baseline** (checkpoint do kit, UNet 5 épocas) | **0.2234** | 0.1117 | Número-base. Fraco até no Maestro2 → muito teto. |

> Próximo: F1 — supervisionado forte (aug + épocas + Dice+CE + arch). Meta: macula_score bem
> acima de 0.22 no val local antes de partir pra generalização (F2).
> ⚠️ Lembrete: melhorar só o macula_score **não** garante leaderboard — o placar real pesa
> doente (0.7), WideField (50%) e penaliza cair no Triton. Por isso o val cross-vendor da F2.

#Tecnologia #Academia
