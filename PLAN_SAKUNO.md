# PLANO — campanha de submissões do Dr. Sakuno (time próprio, 5 balas)

> Para o Claude Code na máquina do Dr. Sakuno (com a RTX 5090). Ele entra como **time próprio**
> (5 submissões + 1 final). Este repo já tem TODO o trabalho do JV. Trabalhe com critério:
> **validar offline no proxy ANTES de gastar cada bala** (foi assim que o JV foi de 8º a 4º).

## 0. O que já sabemos (não repita o que falhou)
O JV competiu e chegou a **4º lugar (0.75)** com este pipeline. Progressão e aprendizados:

| Lever | Veredito |
|---|---|
| Aug geométrica `widefield2` (grid distortion + elástica) | ✅ **GRANDE** — destravou WideField (0.55→0.67) |
| Semi-supervisão (pseudo-labels, descoberta genérica de não rotulados) | ✅ essencial p/ WideField |
| Modelo maior (48-384) | ✅ modesto (+0.02) |
| Resolução 512 | ✅ marginal |
| TTA (h-flip + escalas) | ✅ pequeno mas grátis |
| **Ensemble multi-resolução** | ✅ offline +0.025 Mácula / +0.022 WideField (não testado no leaderboard) |
| **Boundary loss (HausdorffDTLoss)** | ❌ **PIOROU — não use** |

Detalhe completo em `STATUS.md` e `docs/06_log_experimentos.md`. Placar real do JV: Mácula 0.79, WideField 0.70.
Líderes: pooyak 0.82, Ura 0.80, nairul 0.78. Nossa lacuna p/ o topo é **Mácula (precisão de borda/MASD)**.

## 1. Setup (uma vez)
1. `git pull` (fork do Sakuno).
2. Ambiente + dados: ver `RUNBOOK_5090.md` §Passo 0-2 (venv + torch cu128 + dados do Drive em `data/`).
   Se já rodou aqui antes, já está pronto. Confirme `data/starter_kit/app_ingestion/input_data/train/`.
3. `git` autenticado p/ dar push dos resultados no fork do Sakuno.

## 2. Como montar uma submissão (1 comando)
O pipeline (`submission/`) treina IN-CONTAINER no servidor (2h GPU) e é **configurável por env var**:
```bash
# escolhe a config do ensemble (sem editar código):
export DAOCT_ENSEMBLE="384:48,96,192,384"     # single big384 (comprovado ~0.75)
# (vazio = default: 3 membros multi-res 256/384/512 16-128)
bash scripts/package_submission.sh <nome>      # gera submission_<nome>.zip
unzip -l submission_<nome>.zip                 # confira: entrypoint.sh, main.py na RAIZ
```
> ⚠️ O `DAOCT_ENSEMBLE` precisa estar setado **no servidor**, em runtime. O jeito robusto: edite o
> default no topo de `submission/main.py` (a lista `ENSEMBLE`) para a config desejada ANTES de empacotar,
> OU adicione `export DAOCT_ENSEMBLE=...` no `submission/entrypoint.sh`. (O env do seu shell local NÃO
> viaja no zip.) Recomendo editar a lista `ENSEMBLE` em `main.py` por submissão.

## 3. Plano das 5 balas (validar offline antes de cada uma)
Antes de CADA submissão, valide a config no proxy: `scripts/exp_resolution.sh` / `eval_ensemble_proxy.py`
(treina rápido na 5090 e mede plain/cosine/radial). Só suba o que ganhar offline.

1. **Âncora (comprovada):** `ENSEMBLE=[(384,"48,96,192,384")]` (single big384 + widefield2 + semi + TTA).
   Esperado ~0.75. Confirma que o pipeline roda no time do Sakuno e fixa a base.
2. **Ensemble multi-res (default):** 3× {256,384,512} 16-128. Offline deu +0.025/+0.022. Testa o ganho real.
3. **Ensemble + big:** `"384:48,96,192,384 256:16,32,64,128 512:16,32,64,128"` (o provado + 2 diversos).
   Provável melhor combinação (junta capacidade + diversidade).
4. **Refinar o vencedor de #1-3:** ler o breakdown (Mácula vs WideField no leaderboard) e atacar o pior.
   Opções: +1 membro com seed diferente; tunar `--conf_threshold`/`--semi_weight` da semi; img maior no membro big.
5. **Melhor config** (ou guardar p/ a final). A **final** (set, test novo) trava a melhor.

## 4. Diagnóstico (grátis, a cada submissão)
Aba **Results** do CodaBench mostra colunas **Final / Macula / Widefield** por submissão. Use pra decidir:
Mácula baixa → resolução/ensemble/capacidade. WideField baixo → widefield2/semi/generic-discovery.

## 5. Regras / armadilhas (já resolvidas no código)
- `--workers 0` (container tem `/dev/shm`=64MB → workers>0 dá Bus error). Já está no main.py.
- Descoberta de imagens robusta (nunca "0 images → Failed"). Já está.
- Zip com arquivos na RAIZ (não use Finder/Compress). `package_submission.sh` faz certo.
- Treino in-container ≤2h (orçamento dinâmico por membro + fallback no checkpoint embarcado). Já está.
- **Não** versione `data/` nem checkpoints grandes. Resultados vão em `results/` + `STATUS.md`; dê push.
- BSD-2 open-source (já há LICENSE).

## 6. Quando terminar cada submissão
Atualize `docs/06_log_experimentos.md` + `STATUS.md` com o score (Final/Mácula/WideField) e dê **commit + push**.
Assim o JV acompanha do outro lado.
