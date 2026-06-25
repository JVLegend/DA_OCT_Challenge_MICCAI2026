# RUNBOOK — execução autônoma na máquina da RTX 5090

> Você é um Claude Code numa máquina com **RTX 5090 (Blackwell, sm_120)**. O dono (João) abriu este
> repositório e pediu pra você **ler e executar o que for necessário**, de forma autônoma, e salvar
> os resultados de volta no GitHub. Este é o seu plano. Trabalhe com critério de engenharia.

## Objetivo
Treinar o(s) modelo(s) de segmentação de retina OCT do DA-OCT Challenge, avaliar localmente,
gravar resultados/status e **dar push no GitHub**, deixando tudo pronto pro João revisar do outro
computador. Se sobrar tempo/GPU, **já adiantar a próxima rodada melhor**.

## Contexto (leia antes)
- `docs/05_analise_baseline_e_estrategia.md` — fraquezas do baseline + scoring real.
- `docs/07_insights_forum.md` — **importante**: a fase de submissão treina IN-CONTAINER em dados
  novos/maiores (2h GPU); descoberta de imagens tem que ser robusta; zip com arquivos na raiz.
- Baseline a bater: `macula_score = 0.2234` no val local.

## Passo 1 — Dados (Google Drive)
Os dados **não** vêm no git. O João te passa um **link do Google Drive**.
- Salve o link em `data/DATA_URL.txt` (já é gitignored) **ou** exporte `DAOCT_DATA_URL=<link>`.
- Se você não tem o link, **peça ao João** antes de continuar.

## Passo 2 — Rodar tudo (1 comando)
```bash
bash scripts/run_all_5090.sh
```
Esse script é idempotente e faz: ambiente (venv + **torch cu128** + deps) → baixa/descompacta os
dados → checa a GPU → **Round 1** (UNet baseline, 200 épocas, 256px — drop-in) → **Round 2**
(UNet maior 32-256, 384px) → grava `results/`, `STATUS.md` e o log de experimentos → **commit + push**.

Se o push falhar por falta de auth, rode `gh auth login` (ou configure um token) e `git push`.

## Passo 3 — Revisar e decidir (agêntico)
Depois que o script terminar:
1. Leia `STATUS.md` e `results/round*/score.json`. Qual round ganhou? Houve overfit (val caindo)?
2. **Atualize `STATUS.md`** com sua leitura e os próximos passos concretos.

## Passo 4 — Adiantar a próxima rodada (se possível, FAÇA)
O maior ganho real está na **domain adaptation de verdade** (F2), que o baseline não faz. Se a GPU
estiver livre, implemente e rode — é exatamente o que o desafio premia (generalizar p/ Triton/WideField):
- **Semi-supervisão por pseudo-labels:** treine no Maestro2 rotulado, gere pseudo-máscaras pros
  não rotulados (Heidelberg_Spectralis, Zeiss_Cirrus, Topcon_Maestro2_unlabeled) com filtro de
  confiança, e re-treine incluindo as de alta confiança. O hook `--semi` em `train_daoct.py` é stub:
  implemente de verdade num arquivo novo (ex. `scripts/train_daoct_semi.py`) ou estendendo o atual.
- **Augmentation de aparência** já está ligada (gamma/contraste/ruído) — bom ponto de partida p/ robustez cross-vendor.
- Avalie com `bash scripts/eval_local.sh <ckpt>` e **registre no log + STATUS.md**. Commit + push.

## Regras
- **Não** versione `data/` nem checkpoints grandes (o script já cuida; `.gitignore` protege). Pode
  versionar `results/` (relatórios, scores, status e checkpoints pequenos).
- Mantenha a arquitetura do **round1** padrão (drop-in no infer do kit). Archs maiores são avaliadas
  via `scripts/infer_daoct.py` (lê a arch do sidecar `.arch.json`).
- Se algo travar (OOM, NaN), reduza batch ou use `--amp fp16`/`--amp off`, anote no STATUS.md e siga.
- Ao terminar cada rodada: **commit + push** pra o João ver do outro computador.

## Entregáveis (o que o João vai olhar)
- `STATUS.md` — placar atual + próximos passos.
- `results/round1/`, `results/round2/` (e além) — relatórios, scores e checkpoints pequenos.
- `docs/06_log_experimentos.md` — linha por experimento.
