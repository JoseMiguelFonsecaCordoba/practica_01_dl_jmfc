# practica_01_dl_jmfc

Proyecto de clasificación de imágenes CIFAR-10 utilizando **VGG11 + Transfer Learning**, exportación a ONNX, servidor de inferencia Triton y un agente HTTP reutilizable.

Flujo principal:

```text
CIFAR-10
  -> train.py
  -> artifacts/best_model.pth
  -> export_con_onnx.py
  -> Triton Model Repository
  -> Triton en Docker Compose
  -> InferenceAgent
  -> clase, confianza y latencia
```

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

## Evaluación del modelo

La evaluación se realiza sobre las 10,000 imágenes del conjunto de prueba de CIFAR-10.

Desde la carpeta interna del proyecto:

```bash
cd practica_01_dl_jmfc
python evaluate.py
```

Para guardar las métricas en un archivo:

```bash
python evaluate.py 2>&1 | tee reports/evaluation_cuda.txt
```

Para ejecutar la evaluación forzando CPU:

```bash
CUDA_VISIBLE_DEVICES="" python evaluate.py 2>&1 | tee reports/evaluation_cpu.txt
```

El script calcula:

- Test loss.
- Accuracy global.
- Accuracy por clase.
- Precision por clase.
- Recall por clase.
- F1-score por clase.
- Tiempo medio de inferencia.
- Matriz de confusión.

Resultados obtenidos:

```text
Accuracy CUDA: 86.70%
Accuracy CPU:  86.70%
```

La CPU tarda más que CUDA, pero produce la misma calidad de predicción en la evaluación realizada.

Las clases con mejor rendimiento fueron `truck`, `automobile`, `ship`, `horse` y `frog`. Las clases más difíciles fueron `cat`, `dog` y `bird`, principalmente por la similitud visual entre ellas.

## Agente reutilizable de inferencia

El proyecto contiene un agente reutilizable llamado `InferenceAgent`.

El agente recibe una imagen, la prepara con el mismo preprocesamiento utilizado durante la evaluación y la envía mediante HTTP al modelo VGG11 servido por Triton.

```text
Imagen
  -> conversión a RGB
  -> resize a 224 x 224
  -> normalización ImageNet
  -> tensor [1, 3, 224, 224]
  -> petición HTTP a Triton
  -> logits
  -> softmax
  -> clase, confianza y latencia
```

El agente está implementado en:

```text
practica_01_dl_jmfc/agent/inference_agent.py
```

Sus valores predeterminados son:

```text
Servidor Triton: http://localhost:8000
Modelo: vgg11
Entrada: input
Salida: output
```

El agente devuelve:

- `predicted_class`: índice numérico de la clase predicha.
- `class_name`: nombre de la clase CIFAR-10.
- `confidence`: confianza de la predicción.
- `inference_time_ms`: tiempo total de la petición.
- `logits`: salida directa del modelo antes de aplicar softmax.

## Levantar Triton con Docker Compose

Desde la raíz del repositorio:

```bash
docker compose -f practica_01_dl_jmfc/utils/docker-compose.yml up
```

Para ejecutarlo en segundo plano:

```bash
docker compose -f practica_01_dl_jmfc/utils/docker-compose.yml up -d
```

Triton utiliza los siguientes puertos:

- HTTP: `8000`
- gRPC: `8001`
- Métricas: `8002`

Desde otra terminal se puede comprobar el estado del servidor:

```bash
curl -i http://localhost:8000/v2/health/ready
curl -i http://localhost:8000/v2/health/live
curl -i http://localhost:8000/v2/models/vgg11/ready
```

Cada endpoint debe responder con:

```text
HTTP/1.1 200 OK
```

Para detener Triton:

```bash
docker compose -f practica_01_dl_jmfc/utils/docker-compose.yml down
```

## Ejecutar el agente

CIFAR-10 se almacena internamente como un dataset. Para crear temporalmente una imagen PNG de prueba:

```bash
cd practica_01_dl_jmfc

python - <<'PY'
from pathlib import Path
from torchvision.datasets import CIFAR10

dataset = CIFAR10(
    root="data",
    train=False,
    download=False,
    transform=None,
)

image, label = dataset[0]
output_path = Path("/tmp/cifar10_sample.png")
image.save(output_path)

print(f"Imagen guardada en: {output_path}")
print(f"Etiqueta real: {label} - {dataset.classes[label]}")
PY
```

Con Triton ejecutándose, prueba el agente:

```bash
python -m serving.test_inference_agent /tmp/cifar10_sample.png
```

Una salida correcta es similar a:

```text
Inferencia realizada correctamente
Clase predicha: 3
Nombre: cat
Confianza: 0.8229
Tiempo de inferencia: ... ms
Logits: [...]
```

El agente fue validado con una imagen de CIFAR-10 cuya etiqueta real era `cat`. La predicción obtenida fue también `cat`, con una confianza aproximada de `0.8229`.

## Pruebas automáticas

Las pruebas del agente se encuentran en:

```text
practica_01_dl_jmfc/tests/test_inference_agent.py
```

Se pueden ejecutar sin tener Triton encendido:

```bash
cd practica_01_dl_jmfc
python -m unittest discover -s tests -v
```

Las pruebas validan:

- Carga de imágenes.
- Forma del tensor.
- Tipo de datos.
- Imagen inexistente.
- Lectura de logits.
- Respuestas malformadas.
- Cálculo de la clase predicha.
- Cálculo de la confianza.

Resultado esperado:

```text
Ran 5 tests ... OK
```

Las pruebas no necesitan conectarse a Triton porque simulan la respuesta HTTP del servidor.

## Problemas frecuentes

### No existe la imagen

Comprueba que la ruta proporcionada existe:

```bash
ls -l /tmp/cifar10_sample.png
```

### No se pudo conectar con Triton

Comprueba que el contenedor está activo:

```bash
docker ps
curl -i http://localhost:8000/v2/models/vgg11/ready
```

### Triton no encuentra el modelo

Comprueba que existe:

```text
practica_01_dl_jmfc/serving/model_repository/vgg11/1/model.onnx
```

Si no existe, exporta nuevamente el modelo:

```bash
cd practica_01_dl_jmfc
python serving/export_con_onnx.py
```

### Diferencias entre los tiempos de CUDA, CPU y Triton

El tiempo de evaluación local mide principalmente la ejecución del modelo.

El tiempo del agente incluye además:

- Preprocesamiento de la imagen.
- Serialización JSON.
- Comunicación HTTP.
- Ejecución en Triton.
- Lectura de la respuesta.
- Cálculo de softmax.

Por eso los tiempos no deben compararse directamente sin indicar qué parte del flujo se está midiendo.
