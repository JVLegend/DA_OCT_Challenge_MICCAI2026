# Aprendizados do DA-OCT Challenge (MICCAI 2026)

> Autópsia escrita em 13/09/2026, depois do resultado da Final Phase.
> Todos os números vêm do leaderboard da competição e do relatório detalhado da
> submissão 3420 (`detailed_results/3420`). Nada aqui é estimativa.

## Resultado final

| | |
|---|---|
| Submissão final | `submission_round7_tta.zip` (big384 + widefield2 + semi + TTA) |
| ID / data | 3420 · 09/09/2026 16:59 · status Finished |
| **Score final** | **0.7257** |
| Mácula | 0.7586 |
| WideField | 0.6928 |
| **Classificação** | **18º de 22** times com submissão final pontuada (107 inscritos) |

Topo: yzjaxon 0.807 (0.84 / 0.78), UWCO 0.804 (0.83 / 0.78), pooyak 0.803 (0.84 / 0.77).
Dr. Sakuno (`gsakuno`), time próprio: 0.768 (0.81 / 0.73), 15º lugar.

Progressão na fase de submissão: 8º (0.65) > 3º (0.72) > 4º (0.75). A queda para 18º na
final não é regressão do modelo, é o campo tendo continuado a evoluir até 07/09 enquanto
o nosso ficou parado no patamar 0.75.

## 1. A adaptação de domínio funcionou. O que faltou foi base.

Esta é a conclusão mais importante e contraria o que assumimos durante toda a competição.

A métrica que mede se a adaptação de domínio funciona não é o WideField absoluto, é o
**gap entre Mácula e WideField**, ou seja, quanto do desempenho sobrevive à transferência.

| Faixa do leaderboard | Gap médio Mácula menos WideField |
|---|---|
| Top 9 (0.793 a 0.807) | **-0.058** |
| 10º ao 17º (0.748 a 0.780) | **-0.086** |
| **Nós (18º)** | **-0.066** |
| Baseline do organizer | -0.130 |

Nosso gap é de nível de campeão. Está melhor que o de todos os times entre 10º e 17º,
incluindo o do Dr. Sakuno (-0.08). A aug geométrica `widefield2` mais a semi-supervisão
por pseudo-rótulos entregaram exatamente o que deveriam entregar.

O déficit foi uniforme: -0.081 no score final, -0.081 na Mácula e -0.087 no WideField
contra o primeiro colocado. Três buracos do mesmo tamanho. Isso não é um truque que
faltou, é qualidade bruta de segmentação abaixo do necessário em toda a linha.

**Regra prática que sai daqui:** meça sempre nível absoluto e gap separadamente.
Gap grande significa problema de domínio, e aí augmentation e semi-supervisão resolvem.
Gap pequeno com nota baixa, que foi o nosso caso, significa problema de capacidade, e aí
o caminho é arquitetura, resolução e tempo de treino, não mais augmentation.

## 2. Onde o modelo quebrou, por protocolo

Do relatório de piores casos, o quartil inferior de cada protocolo, ordenado pelo **teto**
(a melhor nota dentro das 25% piores). O teto separa falha pontual de buraco estrutural.

| Protocolo | Pior imagem | Teto do pior quartil | Leitura |
|---|---|---|---|
| Maestro2 Macula 6x6 | 0.244 | 0.709 | bom, com falhas pontuais |
| Cirrus Macula | 0.590 | 0.708 | mais robusto |
| Cirrus Macula 6x6 | 0.619 | 0.701 | robusto |
| Triton Macula 6x6 | **0.189** | 0.668 | colapso pontual |
| Spectralis Macula 20x20 | 0.439 | 0.663 | fraco |
| Spectralis Macula | 0.461 | 0.629 | fraco |
| Triton Macula 12x12 | 0.327 | 0.606 | fraco, sem teto |
| **Maestro2 WideField** | 0.373 | **0.543** | **buraco estrutural** |

Três leituras:

**WideField é o único buraco estrutural.** É o único protocolo cujo quartil inferior
inteiro fica abaixo de 0.55. Em todos os outros o quartil ruim ainda se recupera para
0.61 a 0.71. Não são algumas imagens ruins, é o protocolo inteiro puxando.

**Triton é onde a generalização quebra, e ele é penalizado duas vezes.** O vendor não
visto teve os piores casos absolutos (0.189 e 0.327). A fórmula de score subtrai
λ·max(0, média_seen menos média_geral), então cada ponto perdido no Triton entra na média
e ainda aciona a penalidade de generalização.

**Cirrus foi o mais robusto, e ele também era não rotulado.** Isso importa: adaptação que
funciona num domínio alvo não garante nada em outro. Cirrus e Triton estavam na mesma
condição de partida e tiveram destinos opostos. Nunca assuma transferência, meça por
domínio.

## 3. O que aprendemos sobre WideField

Vale isolar, porque é reaproveitável em qualquer trabalho de OCT de campo amplo.

- **WideField não tem rótulo nenhum.** As anotações da competição cobrem só Topcon
  Maestro2 Macula 6x6. WideField é adaptação de domínio pura, sempre.
- **Ainda assim vale metade da nota.** Final = 0.5·Mácula + 0.5·WideField. Metade do
  score vem de um domínio sem ground truth. Isso deveria ter ditado o orçamento de
  esforço desde o primeiro dia.
- **Ninguém resolveu.** O melhor WideField do campo inteiro foi 0.78, contra 0.84 de
  Mácula. Todo time do leaderboard tem gap negativo. O teto é mais baixo para todos.
- **A dificuldade é geométrica.** Campo de visão de 12x9mm contra 6x6mm da mácula
  introduz curvatura retiniana forte, disco óptico e camadas em diagonal íngreme na
  periferia. Nos piores painéis a banda atravessa a imagem inclinada e as camadas finas
  colapsam.
- **O que ataca isso:** augmentation geométrica agressiva. A `widefield2`
  (RandGridDistortion com `num_cells=8`, `distort_limit=0.4`, mais Rand2DElastic com
  `spacing=(30,30)`, `magnitude_range=(2,5)`, mais RandAffine ampliado) levou o WideField
  de 0.55 para 0.67 na fase de submissão. Foi o maior ganho isolado de toda a campanha.

## 4. Levers testados: veredito

| Lever | Veredito |
|---|---|
| Aug geométrica `widefield2` | ✅ **o maior ganho** (WideField 0.55 > 0.67) |
| Semi-supervisão com pseudo-rótulos | ✅ essencial para os domínios sem rótulo |
| Modelo maior (48-384) | ✅ modesto (+0.02) |
| Resolução 512 | ✅ marginal |
| TTA (h-flip + escalas) | ✅ pequeno e barato (+0.01 real, o proxy previa +0.045) |
| Ensemble multi-resolução | ⚠️ +0.025 no proxy, **nunca testado no leaderboard** |
| Boundary loss (HausdorffDTLoss) | ❌ **piorou tudo, não use** |

## 5. Erros de estratégia, que custaram mais que os técnicos

**Gastamos a munição cedo.** As 5 submissões foram usadas até 28/06 e o modelo travou em
0.75. A fase seguiu aberta até 07/09 e o campo subiu para 0.80 enquanto estávamos sem
bala. Parar de iterar num jogo onde ninguém para é o erro mais caro desta competição.
Orçamento de submissões precisa de plano para a janela inteira, com cota reservada para o
fim.

**Construímos a melhoria tarde demais.** O ensemble ficou pronto, ganhava offline, e não
havia mais submissão para validá-lo. Pior: a regra da final exigia o mesmo algoritmo da
melhor entrega anterior, então ele era inutilizável por definição. Antes de investir em
qualquer melhoria grande, pergunte se dá para testá-la no leaderboard dentro do prazo.

**Confiamos num proxy que superestimava.** O proxy de curvatura acertou a direção do
ganho geométrico, mas errou a magnitude em outros levers (previu +0.045 do TTA, entregou
+0.01). Proxy offline serve para ordenar candidatos, não para prever o número real.
Calibre o fator de superestimação assim que houver dois pontos de comparação.

**Diagnosticamos com convicção o alvo errado.** Passamos meses tratando o WideField como
O gargalo. Os números finais mostraram déficit uniforme e um gap de domínio já em nível de
campeão. Estávamos otimizando o que já estava bom.

## 6. O que reaproveitar

- `scripts/train_daoct.py --aug widefield2` é a receita de augmentation geométrica que
  funciona para OCT de campo amplo.
- `scripts/train_daoct_semi.py` com descoberta genérica de não rotulados, para qualquer
  cenário de adaptação de domínio com alvo sem label.
- `scripts/eval_proxy_widefield.py`, a metodologia de validar offline antes de gastar
  submissão. Ela levou o projeto de 8º a 4º. O problema não foi o método, foi ter parado
  de usá-lo quando as balas acabaram.
- Gotchas de container que valem para qualquer submissão compute-to-data: `/dev/shm` de
  64MB exige `--workers 0`, descoberta de imagens tem que ser recursiva e robusta, e o zip
  precisa dos arquivos na raiz.

## Referências

- Código da submissão final: tag git `final-submission-round7-tta` (commit `fa6b82d`)
- Log de experimentos: `docs/06_log_experimentos.md`
- Status da campanha: `STATUS.md`
- Playbook do Dr. Sakuno: `PLAN_SAKUNO.md`
