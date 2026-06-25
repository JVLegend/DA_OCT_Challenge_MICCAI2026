# 04 — Cronograma, Limites e Métricas

## Fases
| Fase | Período | Status | Limites |
|---|---|---|---|
| Pré-Submissão | 26/05 → 22/06/2026 | encerrada | 50 subs (máx 5/dia) · 15 min runtime · **sem GPU** |
| **Submissão** | **23/06 → 07/09/2026** | **ATUAL** | **5 submissões no total** · runtime longo (~2h+) · **com GPU** |
| Final | 07/09 → 14/09/2026 | futura | 1 submissão · resultados em **15/09/2026** |

> [!important] Só 5 submissões na fase atual
> Cada bala conta. Validar localmente (baseline + nossos sanity checks) **antes** de
> gastar submissão. Não fazer "submit pra ver no que dá".

## Métricas de avaliação
> [!note] Fórmula exata extraída do código (`app_scoring/program/metrics.py`).
> Detalhe em [[05_analise_baseline_e_estrategia]] §B. Resumo:
- Score/imagem = média das 10 classes de `0.5·(Dice + exp(−MASD/0.02))`.
- Coorte/vendor = `0.3·saudável + 0.7·doente` (**doente pesa mais**).
- Anatomia = média dos vendors − `0.5·penalidade(unseen)`; **Final = 0.5·Mácula + 0.5·WideField**.
- SEEN = Maestro2/Spectralis/Cirrus · **UNSEEN = Triton** (dispara a penalidade).
- ⚠️ Página pública dizia **λ=1.5**; código do kit usa **0.5**. A fase final pode diferir —
  otimizar generalização ganha nos dois casos.

## Prêmio de Eficiência
Há um prêmio à parte para soluções eficientes (tempo/recursos) — vale manter o modelo
enxuto, não só preciso.

## Licença e ética
- Código aberto sob **BSD 2-clause**.
- Permitido usar dados/modelos **públicos** (ex.: pesos pré-treinados públicos).
- **Proibido** usar dados/anotações privados.

## Datas-chave (resumo)
- ⏳ Fim da fase de Submissão: **07/09/2026**
- 🏁 Submissão final: até **14/09/2026**
- 📣 Resultados: **15/09/2026**

#Academia #Pendente #Tecnologia
