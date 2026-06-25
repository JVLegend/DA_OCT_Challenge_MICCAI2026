# 03 — Protocolo de Submissão (Docker)

> [!warning] Interface única e obrigatória
> O repo baseline define a **única** interface de execução aceita. Toda submissão precisa
> seguir o template à risca. Repo: https://github.com/wusmai/miccai-challenge-daoct-baseline

## Modelo compute-to-data
Você **não baixa** os dados protegidos de avaliação. Empacota treino + inferência num
**container Docker**, sobe, e o servidor seguro constrói e executa perto dos dados.

## Build & Run (do baseline)
```bash
# build
docker build -f ./Dockerfile -t $DOCKER_IMAGE .

# run (monta o diretório atual em /app_ingestion)
docker run -it --rm -v ./:/app_ingestion -w /app_ingestion $DOCKER_IMAGE bash
```
- Ponto de montagem de I/O: **`/app_ingestion`** (working dir).

## Arquivos exigidos no container
- `Dockerfile`
- `requirements.txt`
- `scripts/sanity_check_dataset.py`
- `scripts/train_test_monai.py` — treino **supervisionado**
- `scripts/train_test_monai_semi.py` — treino **semi-supervisionado**
- `scripts/infer_test_monai.py` — **inferência**
- `scripts/metrics.py` — avaliação

## Comandos de referência
```bash
# treino supervisionado (só Topcon rotulado)
python scripts/train_test_monai.py --data_root release_dataset/Topcon_Maestro2

# treino semi-supervisionado (usa também os não rotulados)
python scripts/train_test_monai_semi.py --data_root release_dataset/

# inferência
python scripts/infer_test_monai.py --input_dir <images> --output_dir <predictions> --model_path <checkpoint>
```

## Restrições
- **Imagem-only na inferência**: sem acesso a device type ou disease label.
- Código **open-source sob licença BSD 2-clause**.
- Dados/modelos **públicos** permitidos; **dados privados proibidos**.

#Tecnologia #Academia
