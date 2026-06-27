# STATUS — DA-OCT Challenge

Última atualização: 2026-06-26 (F3+F4 concluídos — pipeline de submissão pronto)

## Estado
- ✅ F0: pipeline local fechado. Baseline: **macula_score = 0.2234**.
- ✅ F1: round1 + round2 (supervisionado forte na 5090).
- ✅ F2: semi-supervisão por pseudo-labels. Round3_semi é o melhor checkpoint.
- ✅ F3: refinamento de bordas (Gaussian one-hot horizontal, +0.0002 macula_score).
- ✅ F3: relatório de confiança cross-vendor (ver `results/crossvendor_confidence.json`).
- ✅ F4: **SUBMETIDO** (ID 3150, 26/06) → **leaderboard Score 0.65** (bala 1/5 usada).
- ⏳ Próximo: F5 (val cross-vendor holdout) + atacar MASD/borda antes da bala #2.

## Resultados (val local = Maestro2/Mácula/saudável — proxy, não leaderboard)

| Round | arch / img | best_val_dice | macula_score | final(local) | vs baseline |
|---|---|---|---|---|---|
| baseline (F0) | UNet 5ep / 256 | — | 0.2234 | 0.1117 | referência |
| round1 (F1) | 16-128 / 256 | 0.8853 | 0.2337 | 0.1169 | +4.6% |
| round2 (F1) | 32-256 / 384 | 0.8853 | 0.2394 | 0.1197 | +7.2% |
| round3_semi (F2) | 32-256 / 384 + pseudo | 0.9019 | 0.2412 | 0.1206 | +8.0% |
| **round3_semi + refine (F3)** | idem + Gaussian one-hot | 0.9019 | **0.2414** | **0.1207** | **+8.1%** |

**Melhor (local): round3_semi + refinamento.**

## 🏁 Leaderboard REAL (1ª submissão)

| Submissão | ID | Score | Detalhe |
|---|---|---|---|
| round3_semi (fix workers=0) | **3150** | **0.65** | Finished. Treino in-container: 437 rotuladas, supervisionado 150ep (val_dice 0.8925) + semi 80ep, pseudo 4293/4367 aceitos, infer 628 imgs +refine +native, 26min total. |

> **Aprendizado-chave:** o val-proxy local (macula 0.24 / final 0.12) **NÃO prevê** o leaderboard.
> Âncora real agora = **0.65**. Toda melhoria futura se mede contra isso (e só via submissão real,
> já que WideField/Triton não existem no val local). Restam **4 balas** (+1 final).

### Breakdown (do leaderboard, 26/06) — onde estamos
| # | Time | Final | Mácula | WideField |
|---|---|---|---|---|
| 1 | pooyak | 0.82 | 0.86 | 0.77 |
| 2 | nairulislam | 0.78 | 0.83 | 0.72 |
| 3 | rickychan2014 | 0.69 | 0.72 | 0.66 |
| 8 | **JV (nós, 3150)** | **0.65** | **0.74** | **0.55** |

**Diagnóstico:** Mácula 0.74 = **3ª melhor** (forte). **WideField 0.55 = gargalo** (líderes 0.72–0.77).
Se WideField → ~0.70, final → ~0.72 = **top 3**. Toda a próxima rodada mira **WideField sem regredir a Mácula**.
Bônus da página: runtime real **≥6h** (usamos 26min!) · λ_penalty real **1.5** · α<0.4 (doente pesa mais).

**Bala #2 (ID 3152 — descoberta genérica de não rotulados):** Final 0.65 · Mácula 0.75 · **WideField 0.55 (inalterado)**.
→ **Hipótese DESCARTADA:** o gargalo do WideField NÃO é dado faltando na semi. Pseudo-rotular wide-field
com teacher treinado só em Mácula não ajuda. WideField = problema de **domínio/geometria** (curvatura do
campo largo que o modelo nunca viu). **2/5 balas usadas.** Próximo: generalização geométrica via augmentation
(elástica + affine forte simulando curvatura do wide-field), **sem precisar de labels de wide-field**.

### Investigação (proxy offline de curvatura) — confirma a hipótese
Wide-field = FOV **12×9mm** (vs Mácula 6×6) → camadas muito mais curvas + disco óptico. Proxy
(`scripts/eval_proxy_widefield.py`) deforma a val de Mácula e mede a queda do round3_semi:
`amp0.12→-4% · amp0.20→-9% · amp0.30→**0.60** (-25%) · amp0.45→0.32`. A curvatura **forte (0.30)**
reproduz a faixa do WideField real (0.55) → **fragilidade geométrica confirmada**. Bala #3 = treinar com
`--aug widefield` (grid distortion + affine grande) e validar no proxy (queda menor + Mácula intacta)
ANTES de submeter. Experimento offline: `scripts/exp_widefield_aug.sh`.

**Resultado do experimento (16-128/256, 100ep):** `aug=widefield` vs `aug=strong` no proxy curvatura 0.30:
plain 0.766 vs 0.770 (**Mácula intacta**) · curvatura **0.567 vs 0.509 (+11%, robustez recuperada)**.
→ Aug geométrica é ganho **líquido positivo e de baixo risco**. Candidata sólida pra bala #3 (modelo cheio
32-256/384 + semi, `--aug widefield`). Ganho esperado no WideField real: **modesto** (proxy ≠ wide-field real,
que tem também disco óptico/aparência). Pode valer 1 iteração offline a mais (aug + elástica) antes da bala.

## Confiança cross-vendor (round3_semi — quanto o modelo generaliza)

| Vendor | N | mean conf | ≥0.85 | ≥0.90 | ≥0.95 |
|---|---|---|---|---|---|
| Heidelberg_Spectralis | 225 | 0.9832 | 100% | 99.6% | 97.3% |
| Zeiss_Cirrus | 212 | 0.9862 | 100% | 100% | 97.6% |
| Topcon_Maestro2_unlabeled | 662 | 0.9856 | 99.8% | 99.5% | 97.7% |

Confiança altíssima em todos os vendors → modelo genuinamente generalizou.
(widefield_score só aparece no leaderboard real — Triton e WideField não estão no val local.)

## Submissão

- **ZIP:** `submission_round3_semi.zip` (arquivos na raiz ✅, entrypoint.sh inalterado ✅)
- **Checkpoint:** round3_semi (UNet 32-256 / 384px / F2 semi) em `submission/checkpoints/`
- **Fluxo no servidor:** warm-start → fine-tune supervisionado (60ep) → semi-supervisão (60ep) → infer + refine + native_size
- **Orçamento:** 95 min treino + ~5 min infer = dentro das 2h ✅

## Próximos passos
1. **Submeter** `submission_round3_semi.zip` (1ª das 5 balas) e aguardar leaderboard.
2. **F5 — val cross-vendor real:** holdout de 1 vendor (Maestro2 Diseased como proxy de domain shift) para medir generalização antes da 2ª bala.
3. **Ajustar** sigma do refinamento ou explorar boundary loss (MASD direto no treino).
4. Revisar resultado leaderboard → decidir próxima melhoria antes de gastar 2ª bala.
