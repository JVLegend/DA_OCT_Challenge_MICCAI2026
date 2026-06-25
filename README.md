---
projeto: DA-OCT Challenge (MICCAI 2026)
status: ativo
inicio: 2026-06-25
fase_atual: Submission (23/06 → 07/09/2026)
---

# 🩻 DA-OCT Challenge — MICCAI 2026

> [!info] Domain Adaptation for solving multivendor retinal OCT dependence in deep learning models
> Desafio satélite do **MICCAI 2026**. Segmentação de **10 camadas da retina** em imagens de **OCT**, generalizando entre diferentes **fabricantes de aparelho** (Topcon, Zeiss, Heidelberg) via *domain adaptation*.

**Link da competição:** https://qtim-challenges.southcentralus.cloudapp.azure.com/competitions/42/#/pages-tab
**Baseline oficial:** https://github.com/wusmai/miccai-challenge-daoct-baseline
**Dataset base:** NIH Bridge2AI **AI-READI** (modelo *compute-to-data*)

---

## TL;DR

- **Tarefa:** segmentação semântica pixel-a-pixel, **10 classes** (8 camadas intrarretinianas + 2 fundos), em B-scans de OCT.
- **O pulo do gato:** só o aparelho **Topcon Maestro2** tem máscaras (labels). Zeiss Cirrus e Heidelberg Spectralis vêm **sem label** → o modelo precisa generalizar (semi/unsupervised domain adaptation).
- **Sem metadados na inferência:** o modelo recebe **só a imagem**. Não sabe o fabricante nem se é olho doente/saudável.
- **Submissão = Docker.** Você não baixa os dados protegidos: sobe um container com seu código de treino+inferência, que roda num servidor seguro.
- **Métricas:** Dice (DSC) + Mean Absolute Surface Distance (MASD), com penalidade de generalização (λ=1.5) para fabricantes não vistos.
- **Prêmios:** US$ 2.000 / 500 / 250 + US$ 500 de eficiência. Top 3 apresenta no MICCAI.

---

## Estrutura da pasta

| Pasta | O que vai aqui |
|---|---|
| `docs/` | Documentação do desafio (visão geral, dados, submissão, cronograma) |
| `data/` | **Dados baixados** (starter kit + example dataset) — *não versionado no git* |
| `baseline/` | Clone do repo baseline oficial (MONAI) |
| `submission/` | Nosso container de submissão (Dockerfile + scripts) |
| `scripts/` | Utilitários nossos (exploração, conversão, sanity checks) |
| `notebooks/` | Exploração e prototipagem |
| `models/` | Checkpoints treinados — *não versionado no git* |

## Documentação

- [[docs/01_visao_geral]] — o que é o desafio, prêmios, parceiros
- [[docs/02_dados_e_labels]] — estrutura dos arquivos, as 10 classes, formato PNG
- [[docs/03_protocolo_submissao]] — interface Docker, build/run, regras de I/O
- [[docs/04_cronograma_e_regras]] — fases, deadlines, limites de submissão, licença

#PhD #Tecnologia #Saude #Academia
