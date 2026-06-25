# 07 — Insights do Fórum (lido em 25/06/2026)

Fonte: fórum oficial (login). Threads mais relevantes resumidas + o que muda na estratégia.

## 🔴 Mudanças de estratégia (o que importa)

### 1. A fase de Submissão tem dados NOVOS e você TREINA no servidor (thread #130, organizer)
> "Will the submission phase release a new training and validation dataset? — **yes**"
> "Do I need to train server-side or just inference? — **you should still train before inference
> as there is more to train on to make your model better**"
- O example dataset que temos (230 Maestro2 rotulados) é **proxy**. O servidor tem **mais dados**.
- O fluxo pretendido: o container **treina** (na data montada, maior) **e depois infere**, dentro
  das **2h** com GPU. Mandar só um checkpoint treinado no exemplo pequeno **subaproveita** os dados.
- **Ação:** `train_daoct.py` precisa caber **train+infer < 2h** na GPU do servidor. Decidir: treinar
  in-container (sem checkpoint) vs. checkpoint + fine-tune. Adicionar teto de tempo/épocas.

### 2. Fase Final treina de novo (thread #130)
> "final phase: 1 submission on a new val testing set, **same training set** as submission phase…
> we assume participants will **train during this phase too** and reproduce a good model."
- Final = 1 bala, val novo, **mesmo train**, treina in-container. **Determinismo/reprodutibilidade importam.**

### 3. Gotcha nº1: "rodou nas 173 mas deu Failed" = 0 imagens descobertas (thread #120)
Log real de um participante mostra paths do servidor **diferentes do kit local**:
```
Using DATA_ROOT: /app/input/data
Using INPUT_DIR:  /app/input/ref
Using OUTPUT_DIR: /app/output
[INFO] Discovered 0 images for inference.   <-- mata o score
```
- No servidor os dados são montados em `/app/input/...` (não no `/app/input_data/` do kit local).
- **Ação (F4):** a descoberta de imagens da inferência tem que ser **recursiva e robusta** (nunca
  retornar 0). Logar a contagem. Não depender de uma subpasta fixa (`val/images`).

### 4. Empacotamento do ZIP: arquivos na RAIZ (thread #118)
- Os arquivos têm que ficar na **raiz** do zip, não dentro de uma subpasta. Finder (Compress) do
  macOS embrulha numa pasta extra → **falha**.
- **Correto:** `cd program && zip -r ../submission.zip .`  (o `.` = conteúdo na raiz)
- Conferir antes de submeter: abrir o zip e ver `entrypoint.sh`, `main.py`, etc. **na raiz**.

### 5. Template/regras de submissão (thread #114, organizer)
- Starting kit fica na aba **Files** (esquerda na home do desafio). Tem sample submission.
- **NÃO alterar `entrypoint.sh`.** Precisa ter um `main.py` que o entrypoint chama e passa argumentos.

## 🟡 Orçamento de submissões — cuidado dobrado
- **5 no total** na fase de Submissão; depois acabou até a Final (#114, #130).
- "**be careful and maybe do a lot of review of results from first run before the second**" (#130).
- **Falhas:** thread #134 — pergunta "failed conta?" → organizer respondeu só "**Yes**" (ambíguo:
  pode ser "sim, é igual ao pré-sub onde NÃO contavam" OU "sim, contam"). **Não confiar** que falha
  não conta — tratar toda submissão como bala gasta.
- **Canceladas:** thread #135 — submissão cancelada por reboot do sistema; **sem resposta** ainda.
- Plataforma instável: várias threads de auto-cancel / 500 / "stuck preparing" (#119,#126,#131,#132).
  Operar com margem: validar local exaustivamente, submeter com folga de tempo, reter logs.

## 🟢 Ambiente de execução (config + #114/#115)
- Pré-Submissão (encerrada): curto, **CPU-only** ("NVIDIA Driver not detected"), 5/dia, 50 total.
- **Submissão (atual): GPU habilitada, 2h/submissão, 5 no total.**
- Final: 2h, 1 submissão.
- Tamanho do held-out: perguntado (#115) mas sem resposta confirmada; val local tem 173.

## Observação
A conta logada (`joaodias`) aparece com "Admin management"/"Queue Management"/"Server Status" nessa
instância Codabench — se isso for acesso real de organizador/admin, dá pra confirmar direto o
comportamento de quota (falha/cancelada conta?) em vez de inferir do fórum.

## Itens de ação (entram no plano)
- [ ] F4: endurecer descoberta de imagens (recursiva, multi-padrão, loga contagem, nunca 0).
- [ ] Ajustar pipeline para **treinar in-container < 2h** na GPU do servidor (teto de épocas/tempo).
- [ ] Script de empacotamento que zipa com arquivos na **raiz** (sem Finder).
- [ ] Garantir determinismo (seed) para a fase Final.
- [ ] Tratar toda submissão como bala; revisar resultados a fundo entre uma e outra.

#Tecnologia #Academia #Saude
