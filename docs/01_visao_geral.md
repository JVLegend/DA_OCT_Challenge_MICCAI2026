# 01 — Visão Geral

## O problema clínico/técnico
Modelos de deep learning para OCT (tomografia de coerência óptica) da retina sofrem de
**dependência de fabricante**: um modelo treinado em imagens de um aparelho (ex.: Topcon)
degrada quando recebe imagens de outro (Zeiss, Heidelberg), por diferenças de física do
sinal, ruído e protocolo de aquisição.

O **DA-OCT Challenge** pede um modelo que produza **representações invariantes a fabricante
e a protocolo** — ou seja, que segmente as camadas da retina igualmente bem em aparelhos
que ele **nunca viu rotulados**.

## Números (em 25/06/2026)
- 56 times inscritos · 250 submissões já feitas
- Fase atual: **Submission** (aberta até 07/09/2026)

## Prêmios
| Colocação | Prêmio |
|---|---|
| 1º lugar | US$ 2.000 |
| 2º lugar | US$ 500 |
| 3º lugar | US$ 250 |
| Prêmio de Eficiência | US$ 500 |

Os **3 primeiros** são convidados a apresentar no **MICCAI 2026**.

## Parceiros / origem dos dados
- Desafio satélite do **MICCAI 2026**.
- Dados do **NIH Bridge2AI — AI-READI** (dataset multimodal para pesquisa de olho diabético).
- Modelo **compute-to-data**: os dados protegidos não são baixados; o código roda perto dos dados.

## Por que é estratégico (contexto JV)
Encaixa direto na tese de oftalmologia/IA (AutoRefratorGeek, MedScavador Oftalmo,
GeekVision, Oculômica/Regatieri). Um bom resultado num desafio MICCAI = ativo de
reputação acadêmica (PhD) + validação técnica de segmentação de retina reaproveitável
nos produtos. Ver [[../README]].

#Academia #Saude #Tecnologia #PhD
