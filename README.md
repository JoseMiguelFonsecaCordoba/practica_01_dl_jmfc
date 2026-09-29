# practica_01_dl_jmfc

Primera etapa de producción (CPU) para MNIST con **VGG11 + Transfer Learning**.

Flujo:

`MNIST -> train.py -> artifacts/best_model.pth -> export_con_onnx.py -> Triton Model Repository -> Triton (Docker Compose) -> health checks HTTP`

## Prerrequisitos

- Python 3.10+ (CPU)
- Docker + Docker Compose
- (Primer entrenamiento) Internet para descargar pesos ImageNet de VGG11

## Instalación de dependencias

Desde la raíz del repo:

```bash
python -m pip install -r requirements.txt
```

## Entrenamiento (CPU)

```bash
cd practica_01_dl_jmfc
python train.py
```

Esto genera `artifacts/best_model.pth`.

## Inferencia local mínima (CPU)

```bash
python serving/inferencia_local.py
```

Valida que el modelo cargue y produzca logits con forma `(1, 10)`.

## Exportación a ONNX (CPU)

```bash
python serving/export_con_onnx.py
```

Esto genera `serving/model_repository/vgg11/1/model.onnx`, valida con `onnx.checker` y compara salida ONNX Runtime vs PyTorch.

## Levantar Triton con Docker Compose (CPU)

```bash
docker compose -f utils/docker-compose.yml up
```

Puertos:
- HTTP: `8000`
- gRPC: `8001`
- Métricas: `8002`

> Si falta `serving/model_repository/vgg11/1/model.onnx`, el contenedor falla con un mensaje claro indicando ejecutar `python serving/export_con_onnx.py`.

## Comprobaciones HTTP básicas

En otra terminal:

```bash
./test.bash
```

Opcional con host remoto:

```bash
BASE_URL=http://<IP_DEL_SERVIDOR>:8000 ./test.bash
```

## Detener Triton

```bash
docker compose -f utils/docker-compose.yml down
```
