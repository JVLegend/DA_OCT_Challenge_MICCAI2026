# 05 — Análise do Baseline e Estratégia

> Derivado da **leitura do código** do starter kit (não da página pública). Onde houver
> divergência, o que vale é o código de scoring que roda no servidor.

## A. Como o baseline funciona (e por que é fraco)

Fluxo: `entrypoint.sh` → `main.py` → procura checkpoint → se não achar, treina
`train_test_monai_semi.py` → roda `infer_test_monai.py` → escreve `<case>-mask.png` em `/app/output/`.

Fraquezas do baseline (= nossas oportunidades):
1. **Não faz domain adaptation.** O `train_test_monai_semi.py` carrega os não rotulados mas
   **só treina supervisionado no Topcon Maestro2**. Os unlabeled só passam num "test loop"
   no fim (linhas 160-169). O miolo do desafio está vazio.
2. **Treino de brinquedo:** `epochs = 5  # quick test`, UNet minúsculo `channels=(16,32,64,128)`,
   `lr=1e-3`, **só DiceLoss**, **sem augmentation**.
3. **Resize destrutivo:** tudo vira 256×256, esmagando aspect ratios (Spectralis 1536×384 = 4:1
   vira 1:1). Perde geometria das camadas → ruim pro MASD.
4. **Sem validação cross-vendor:** avalia Dice só no próprio treino Maestro2.
5. Vem um checkpoint pronto `unet_maestro2_semi.pth` (1.6 MB) → dá pra rodar inferência já.

## B. O scoring REAL (de `metrics.py` + `scoring.py`)

Constantes: `NUM_CLASSES=10`, `ALPHA=0.3`, `LAMBDA_PENALTY=0.5`, `TAU=0.02`,
`BETA_MACULA=0.5`, `BETA_WIDEFIELD=0.5`.
`SEEN_VENDORS = [Maestro2, Spectralis, Cirrus]` · `UNSEEN_VENDORS = [Triton]`.

- **Score por imagem** = média sobre as 10 classes de `0.5·(Dice_c + exp(−MASD_c/0.02))`,
  com MASD normalizado pela altura da imagem. τ=0.02 é severo → **borda de camada precisa importa muito**.
- **Coorte por vendor** = `0.3·saudável + 0.7·doente` → **doente pesa mais**.
- **Score por anatomia** = média dos vendors − `0.5 · penalidade`, onde
  `penalidade = média_unseen( max(0, média_seen − score_unseen) )`. Ou seja: cair no **Triton** dói.
- **Final** = `0.5·Mácula + 0.5·WideField`.

> [!important] Implicações diretas pra estratégia
> - Generalizar pra **Triton** (nunca visto) e mandar bem em **WideField** vale metade do jogo.
> - **Doente > saudável** no peso. Não dá pra otimizar só no saudável fácil.
> - **Spectralis e Cirrus são SEEN no placar, mas vêm SEM label no treino** → o uso dos
>   não rotulados (semi-supervisão/DA) é o que captura esses pontos.
> - Divergência: página dizia λ=1.5; código diz 0.5. A fase final pode mudar a config —
>   de qualquer modo, otimizar generalização ganha nos dois casos.

## C. O val set local NÃO é o placar real
Os 173 de `val/images` (+ gabarito em `app_scoring/.../masks`) são **100% Maestro2 / Mácula /
saudáveis**. Serve só pra fechar o loop técnico. Para medir o que o placar mede, **construir
val interno** a partir do treino: segurar parte do Maestro2 rotulado + simular "unseen"
deixando um vendor de fora.

## D. Dados (example dataset) e dimensões
| Domínio | Imagens | Máscara | Dim. nativa (exemplo) |
|---|---|---|---|
| Topcon_Maestro2 | 230 | ✅ | 256×336 |
| Topcon_Maestro2_unlabeled | 662 | ❌ | 256×384 |
| Heidelberg_Spectralis | 225 | ❌ | **1536×384** |
| Zeiss_Cirrus | 212 | ❌ | 768×368 |
| val/images | 173 | (ref) | Maestro2 |

Máscaras: `uint8`, valores **0–9** (10 classes, sem `//10`). Dimensões variam por vendor →
tratar aspect ratio (padding / resize por vendor / patches), não esmagar pra 256².

## E. Estratégia (pilares)
1. **Baseline forte supervisionado (Maestro2):** UNet maior ou encoder pré-treinado público
   (permitido), augmentation pesada, mais épocas, loss Dice+CE (+ boundary/surface loss pro MASD),
   split treino/val próprio.
2. **Generalização por augmentation de aparência:** simular variação de vendor (gamma, contraste,
   ruído, blur, CLAHE) — o jeito mais barato e robusto de cobrir Spectralis/Cirrus/Triton.
3. **Semi-supervisão real com os não rotulados:** pseudo-labels / self-training → depois
   consistency (Mean Teacher) ou alinhamento de features (adversarial/CORAL). Começar simples.
4. **Geometria pro MASD:** preservar aspect ratio, inferir em resolução maior, pós-processar
   bordas de camada (suavização monotônica das camadas retinianas).
5. **Val interno tipo-placar:** holdout Maestro2 + deixar 1 vendor de fora como proxy de "unseen".
6. **Prêmio de eficiência:** manter o modelo enxuto (UNet é ótimo nisso).

## F. Plano de execução (fases)
- **F0 — Ambiente + loop fechado:** venv Python 3.11 (ou Docker oficial), instalar torch+monai+cv2,
  rodar inferência com o checkpoint dado → scoring local nos 173 → **número-base**. Prova que o
  pipeline inteiro roda antes de tocar no modelo.
- **F1 — Supervisionado forte:** bater o baseline no val Maestro2 (aug + épocas + loss + arch).
- **F2 — Domain generalization + semi:** aug de aparência + pseudo-labels nos não rotulados;
  montar val cross-vendor.
- **F3 — MASD/borda + resolução:** otimizar fronteiras de camada e inferência em alta res.
- **F4 — Empacotar e submeter:** zip de submissão, testar via container ingestion+scoring,
  **gastar 1 das 5 balas** só com ganho validado localmente.

## G. Compute
Treino real precisa de **GPU**. Opções do JV: **DGX** (túnel Cloudflare), **Kaggle T4**, ou
Mac CPU (só pra testar loop). Mac Apple Silicon roda a imagem `projectmonai/monai` (linux/amd64)
emulada — ok pra inferência/loop, lento pra treino.

#Tecnologia #Academia #Saude
