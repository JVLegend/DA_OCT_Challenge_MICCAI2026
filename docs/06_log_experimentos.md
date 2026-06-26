# 06 — Log de Experimentos

Comparação entre runs no **val local** (`scripts/run_baseline_local.sh`).
⚠️ Val local = 173 imagens **Maestro2 / Mácula / saudável**. WideField ausente (conta 0).
Não é a nota do leaderboard — serve pra comparar experimentos entre si. A coluna que importa
pra comparar é **macula_score** (desempenho no Maestro2 fácil).

| Data | Experimento | macula_score | final (local) | Notas |
|---|---|---|---|---|
| 2026-06-25 | **F0 — baseline** (checkpoint do kit, UNet 5 épocas) | **0.2234** | 0.1117 | Número-base. Fraco até no Maestro2 → muito teto. |
| 2026-06-26 | **F1 round1** — UNet 16-128, 200ep, 256px, aug strong, bf16 | **0.2337** | 0.1169 | +4.6% vs baseline. best_val_dice=0.8853. |
| 2026-06-26 | **F1 round2** — UNet 32-256, 250ep, 384px, aug strong, bf16 | **0.2394** | 0.1197 | +7.2% vs baseline. |
| 2026-06-26 | **F2 round3_semi** — pseudo-labels 1095/1099 (conf≥0.85), warm-start round2, 150ep, 4.8min | **0.2412** | 0.1206 | +8.0% vs baseline. best_val_dice=0.9019. |
| 2026-06-26 | **F3 round3_semi+refine** — Gaussian one-hot horizontal (σ=2.5px) no pós-processamento | **0.2414** | 0.1207 | +8.1% vs baseline. Melhor até agora. ZIP pronto: submission_round3_semi.zip |

## 🏁 Leaderboard real (submissões)

| Data | ID | Submissão | **Score real** | Notas |
|---|---|---|---|---|
| 2026-06-26 | 3150 | round3_semi (fix workers=0) | **0.65** | 1ª bala. Treino in-container 437 img (150ep sup val_dice 0.8925 + 80ep semi, 4293/4367 pseudo) + infer 628 +refine +native, 26min. |

> ⚠️ **O val-proxy local não previu o leaderboard:** macula 0.24 / final 0.12 local → **0.65 real**.
> O placar real inclui WideField (50%), Triton (penalidade) e MASD severo — nada disso aparece no
> val local. **Âncora = 0.65; restam 4 balas (+1 final).** Próxima melhoria mira MASD/borda e
> generalização, medida idealmente num val cross-vendor (F5) antes de gastar bala.

#Tecnologia #Academia
