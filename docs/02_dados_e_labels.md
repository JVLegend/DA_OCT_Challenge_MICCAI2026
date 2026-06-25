# 02 — Dados e Labels

## Origem
B-scans extraídos do protocolo **Topcon Maestro2 OCTA Macula 6×6 mm**, mais imagens de
outros fabricantes para o domain adaptation. Base: **AI-READI** (NIH Bridge2AI).

## Formato dos arquivos
- Imagens e máscaras em **PNG**.
- Convenção de nome: `xxx-image.png` (imagem) e `xxx-mask.png` (máscara).
- A máscara é **integer-encoded por pixel**: cada pixel tem um valor inteiro de **0 a 9**
  indicando a classe.

## Estrutura de diretórios (após descompactar)
```
release_dataset/
├── Topcon_Maestro2/                 ← COM máscaras (fonte rotulada / supervisão)
│   ├── Diseased/
│   └── Healthy/
├── Topcon_Maestro2_unlabeled/       ← Topcon sem máscara (semi-supervisão)
│   ├── Diseased/
│   └── Healthy/
├── Heidelberg_Spectralis/           ← SEM máscara (domínio alvo)
│   ├── Diseased/
│   └── Healthy/
└── Zeiss_Cirrus/                    ← SEM máscara (domínio alvo)
    ├── Diseased/
    └── Healthy/
```

> [!important] Só o Topcon Maestro2 vem rotulado
> Todo o resto (Zeiss Cirrus, Heidelberg Spectralis, wide-field) é **não rotulado**.
> O desafio é generalizar para esses domínios → semi-supervised / unsupervised
> domain adaptation.

## As 10 classes
**8 camadas intrarretinianas** delimitadas por **2 zonas de fundo**:
- 1 zona cobrindo todo o espaço **pré-retiniano** (acima da retina)
- 1 zona cobrindo o espaço **coroidal/escleral profundo** (abaixo da retina)

Valores de máscara: `0..9`. (Os nomes exatos das 8 camadas saem do `sanity_check_dataset.py`
e da doc do baseline depois que os dados estiverem na pasta `data/`.)

## Coortes
Os dados são estratificados em **Healthy** vs **Diseased**. A avaliação reporta as duas
coortes separadamente (ver [[04_cronograma_e_regras]] → métricas).

## Regra de ouro na inferência
Nenhum metadado é dado no momento da predição: **sem device label, sem disease label**.
A entrada é **apenas a imagem**. Modelos não podem depender de saber o fabricante.

#Saude #Tecnologia #Academia
