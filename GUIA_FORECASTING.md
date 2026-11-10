# Guía de trabajo: forecasting con RNN, LSTM y MLflow

Esta guía describe el flujo habitual para trabajar con la fase de
forecasting del proyecto desde WSL.

## 1. Abrir el proyecto

```bash
cd ~/practica_01_dl_jmfc
code .
```

Actualizar la copia local antes de trabajar:

```bash
git pull origin main
```

## 2. Activar el entorno virtual

Cada nueva terminal de WSL requiere activar el entorno:

```bash
cd ~/practica_01_dl_jmfc
source .venv/bin/activate
```

La terminal debe mostrar algo parecido a:

```text
(.venv) usuario@equipo:~/practica_01_dl_jmfc$
```

Si el entorno todavía no existe, créalo:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r forecasting/requirements.txt
python -m pip install pre-commit
pre-commit install
```

## 3. Crear carpetas persistentes

Ejecuta una vez, o si alguna carpeta fue eliminada:

```bash
mkdir -p forecasting/data
mkdir -p forecasting/artifacts
mkdir -p forecasting/reports
mkdir -p mlflow-data
mkdir -p mlruns
```

Si Docker creó archivos como `root`, corrige sus permisos:

```bash
sudo chown -R "$(id -u):$(id -g)" \
  forecasting/data \
  forecasting/artifacts \
  forecasting/reports \
  mlflow-data \
  mlruns
```

## 4. Validar el código antes de ejecutar

Pruebas unitarias:

```bash
python -m pytest forecasting/tests
```

Comprobar sintaxis:

```bash
python -m py_compile $(find forecasting -name '*.py')
```

Ejecutar todos los hooks:

```bash
pre-commit run --all-files
```

El resultado esperado incluye:

```text
2 passed
ruff Passed
ruff-format Passed
check yaml Passed
```

No usar `git add .` sin revisar antes los archivos, porque el repositorio
también contiene las fases anteriores del proyecto.

## 5. Configurar DVC

DVC versiona datasets y checkpoints grandes sin subirlos directamente a
GitHub. La configuración de DVC está en:

```text
.dvc/
dvc.yaml
```

Instalar DVC si todavía no está disponible:

```bash
python -m pip install dvc
```

Preparar el dataset mediante DVC:

```bash
dvc repro prepare-data
```

Esto descarga Jena Climate en:

```text
forecasting/data/jena_climate_2009_2016.csv
```

Entrenar mediante la etapa reproducible de DVC:

```bash
dvc repro train
```

La etapa `train` ejecuta el entrenamiento dentro de Docker y genera los
checkpoints en `forecasting/artifacts/`.

Consultar el estado:

```bash
dvc status
```

Después de generar o cambiar datos y modelos, guardar sus metadatos:

```bash
dvc add -f forecasting/data/jena_climate_2009_2016.csv
dvc add -f forecasting/artifacts
git add forecasting/data/jena_climate_2009_2016.csv.dvc \
  forecasting/artifacts.dvc dvc.yaml dvc.lock .dvc
```

Para un almacenamiento remoto de DVC, configura el destino proporcionado por
el curso antes de ejecutar `dvc push`:

```bash
dvc remote add -d storage <URL_DEL_ALMACENAMIENTO>
dvc push
```

No subas el CSV ni los checkpoints directamente con Git.

## 6. Construir la imagen de entrenamiento

Construir la imagen una primera vez:

```bash
docker compose \
  -f docker-compose.forecasting.yml \
  build forecasting-train
```

Reconstruir ignorando la caché después de cambiar el Dockerfile o las
dependencias:

```bash
docker compose \
  -f docker-compose.forecasting.yml \
  build --no-cache forecasting-train
```

## 7. Entrenar RNN y LSTM con GPU

Verificar que Docker reconoce la GPU:

```bash
docker run --rm --gpus all \
  pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
  python -c "import torch; print(torch.cuda.is_available())"
```

La salida esperada es:

```text
True
```

Ejecutar ambos entrenamientos:

```bash
docker compose \
  -f docker-compose.forecasting.yml \
  -f docker-compose.forecasting.gpu.yml \
  run --rm forecasting-train
```

El comando:

- descarga Jena Climate si todavía no existe;
- prepara las ventanas de 24 horas;
- entrena la RNN;
- entrena la LSTM;
- registra parámetros y métricas en MLflow;
- guarda los checkpoints;
- muestra cuál obtuvo menor `val_loss`.

Los checkpoints se guardan en:

```text
forecasting/artifacts/rnn_best.pt
forecasting/artifacts/lstm_best.pt
```

## 8. Entrenar sin GPU

Si no se dispone de GPU:

```bash
docker compose \
  -f docker-compose.forecasting.yml \
  run --rm forecasting-train
```

El entrenamiento será más lento, pero el flujo es el mismo.

## 9. Revisar MLflow

MLflow utiliza SQLite para el seguimiento:

```text
mlflow-data/mlflow.db
```

Los artefactos registrados se guardan en:

```text
mlruns/
```

Iniciar la interfaz de MLflow:

```bash
mlflow ui \
  --backend-store-uri sqlite:///mlflow-data/mlflow.db \
  --host 127.0.0.1 \
  --port 5003
```

Abrir en el navegador:

```text
http://127.0.0.1:5003
```

Para detener MLflow:

```text
Ctrl+C
```

## 10. Seleccionar el mejor modelo

Después del entrenamiento:

```bash
python -m forecasting.select_best
```

El script compara los `val_loss` registrados en el archivo:

```text
forecasting/artifacts/model_metadata.json
```

El modelo seleccionado se copia como:

```text
forecasting/artifacts/best_model.pt
```

## 11. Evaluar ambos modelos

Ejecutar:

```bash
python -m forecasting.evaluate
```

Los resultados se guardan en:

```text
forecasting/reports/evaluation.json
forecasting/reports/rnn_forecast.png
forecasting/reports/lstm_forecast.png
```

Ver el resultado numérico:

```bash
cat forecasting/reports/evaluation.json
```

La métrica principal es `test_rmse_celsius`. El menor valor representa el
mejor desempeño sobre el conjunto de prueba.

## 12. Levantar el servidor de inferencia

El servidor usa el archivo:

```text
forecasting/artifacts/best_model.pt
```

Construir y levantar el contenedor en segundo plano:

```bash
docker compose \
  -f docker-compose.forecasting.yml \
  up -d --build forecasting-inference
```

Comprobar el contenedor:

```bash
docker compose \
  -f docker-compose.forecasting.yml \
  ps
```

Comprobar el endpoint de salud:

```bash
curl http://127.0.0.1:5002/health
```

Respuesta esperada:

```json
{
  "status": "ok",
  "model": "lstm",
  "device": "cpu",
  "sequence_length": 24
}
```

## 13. Probar una predicción

El endpoint requiere exactamente 24 temperaturas:

```bash
curl -X POST http://127.0.0.1:5002/predict \
  -H 'Content-Type: application/json' \
  -d '{"values":[10,11,12,13,14,15,16,17,18,19,20,19,18,17,16,15,14,13,12,11,10,9,10,11]}'
```

Respuesta esperada:

```json
{
  "prediction_celsius": 15.2,
  "model": "lstm",
  "input_points": 24
}
```

El valor exacto puede cambiar según el entrenamiento.

Consultar información del modelo:

```bash
curl http://127.0.0.1:5002/model
```

Ver los logs:

```bash
docker compose \
  -f docker-compose.forecasting.yml \
  logs -f forecasting-inference
```

## 14. Ejecutar las tres pruebas oficiales

Con el servidor levantado, ejecutar:

```bash
bash forecasting/tests/curl_tests.sh
```

También se puede probar un servidor remoto:

```bash
BASE_URL=http://<IP_DEL_SERVIDOR>:5002 \
  bash forecasting/tests/curl_tests.sh
```

Las tres pruebas comprueban:

1. Estado de salud del servicio.
2. Predicción con exactamente 24 valores.
3. Rechazo de una entrada inválida con HTTP 422.

## 15. Detener los contenedores

Detener el servidor de inferencia:

```bash
docker compose \
  -f docker-compose.forecasting.yml \
  down
```

El entrenamiento se ejecuta con `run --rm`, por lo que su contenedor se
elimina automáticamente al terminar.

## 16. Flujo completo recomendado

Para una sesión normal:

```bash
cd ~/practica_01_dl_jmfc
source .venv/bin/activate
git pull origin main

python -m pytest forecasting/tests
pre-commit run --all-files

docker compose \
  -f docker-compose.forecasting.yml \
  -f docker-compose.forecasting.gpu.yml \
  run --rm forecasting-train

python -m forecasting.select_best
python -m forecasting.evaluate

docker compose \
  -f docker-compose.forecasting.yml \
  up -d --build forecasting-inference

curl http://127.0.0.1:5002/health
```

## 17. Flujo para detener todo al terminar

```bash
docker compose \
  -f docker-compose.forecasting.yml \
  down
```

Si MLflow está abierto en otra terminal, detenerlo con:

```text
Ctrl+C
```

## 18. Comprobar Git antes de hacer commit

Revisar el estado:

```bash
git status --short
```

Agregar únicamente archivos de la fase de forecasting:

```bash
git add \
  README.md \
  GUIA_FORECASTING.md \
  .pre-commit-config.yaml \
  pytest.ini \
  Dockerfile.forecasting \
  docker-compose.forecasting.yml \
  docker-compose.forecasting.gpu.yml \
  dvc.yaml \
  .dvc \
  forecasting
```

Revisar lo preparado:

```bash
git diff --cached --stat
git diff --cached --check
```

Crear y subir el commit:

```bash
git commit -m "Document forecasting workflow" \
  -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push origin main
```

No subir:

```text
forecasting/data/*.csv
forecasting/data/*.zip
forecasting/artifacts/*.pt
forecasting/reports/*.png
mlflow-data/
mlruns/
```

Estos archivos están excluidos mediante `.gitignore`.

## 19. Verificación automática en GitHub

El repositorio contiene el workflow:

```text
.github/workflows/forecasting-ci.yml
```

GitHub Actions lo ejecuta automáticamente en cada `push` a `main` y en cada
Pull Request hacia `main`. El workflow:

- instala Python y las dependencias;
- ejecuta las pruebas unitarias;
- comprueba la sintaxis de `forecasting/`;
- ejecuta `pre-commit`.

El workflow valida el código, pero no entrena modelos ni requiere GPU.
Después de hacer `git push`, el resultado puede consultarse en la pestaña
`Actions` del repositorio de GitHub.

## 20. Problemas frecuentes

### `ModuleNotFoundError: No module named 'forecasting'`

Ejecutar desde la raíz del repositorio:

```bash
cd ~/practica_01_dl_jmfc
python -m pytest forecasting/tests
```

El archivo `pytest.ini` configura el `pythonpath`.

### MLflow indica que el file store está en maintenance mode

Actualizar el repositorio:

```bash
git pull origin main
```

La configuración correcta utiliza:

```text
sqlite:////workspace/mlflow-data/mlflow.db
```

### Docker utiliza código anterior

Reconstruir la imagen:

```bash
docker compose \
  -f docker-compose.forecasting.yml \
  -f docker-compose.forecasting.gpu.yml \
  build --no-cache forecasting-train
```

### `PermissionError` en `forecasting/artifacts`

Corregir permisos:

```bash
sudo chown -R "$(id -u):$(id -g)" \
  forecasting/data \
  forecasting/artifacts \
  forecasting/reports \
  mlflow-data \
  mlruns
```

### `curl` no responde en el puerto 5002

Comprobar si el contenedor está activo:

```bash
docker compose \
  -f docker-compose.forecasting.yml \
  ps
```

Levantarlo:

```bash
docker compose \
  -f docker-compose.forecasting.yml \
  up -d forecasting-inference
```

Revisar los logs:

```bash
docker compose \
  -f docker-compose.forecasting.yml \
  logs --tail=100 forecasting-inference
```
