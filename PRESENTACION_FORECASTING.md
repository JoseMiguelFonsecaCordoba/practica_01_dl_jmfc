# Guía de presentación: forecasting con RNN y LSTM

Esta guía contiene el procedimiento recomendado para presentar el proyecto,
una explicación técnica de principio a fin y respuestas breves para preguntas
posibles del profesor.

## 1. Qué está terminado

La fase de forecasting está implementada y validada:

- Dataset Jena Climate.
- Preprocesamiento reproducible y normalización sin fuga de información.
- Ventanas de 24 valores para predecir el siguiente valor.
- Modelos RNN y LSTM implementados en PyTorch.
- Comparación y seguimiento con MLflow.
- Entrenamiento reproducible mediante Docker y DVC.
- Selección automática del mejor modelo.
- Servicio FastAPI en el puerto `5002`.
- Pruebas unitarias y pruebas funcionales con `curl`.
- Validaciones con Ruff, pre-commit y GitHub Actions.
- Informe IEEE en `forecasting/informe_ieee.tex`.

El último workflow de GitHub Actions terminó correctamente. Las ejecuciones
fallidas anteriores corresponden a commits históricos; no indican un fallo del
estado actual.

## 2. Qué falta antes de entregar

No falta una funcionalidad obligatoria. Conviene completar estas tareas de
presentación:

1. Compilar `forecasting/informe_ieee.tex` en Overleaf.
2. Guardar capturas de `/docs`, MLflow, la terminal y GitHub Actions.
3. Confirmar que los artefactos locales del modelo existen en el equipo donde
   se hará la demostración.
4. No volver a entrenar el día de la presentación salvo que sea necesario.

Los checkpoints, reportes y la base SQLite de MLflow están ignorados por Git.
Esto evita subir archivos grandes o generados, pero significa que el equipo de
presentación debe conservar su copia local. Si los artefactos no existen, hay
que ejecutar el entrenamiento antes de la presentación, no durante ella.

## 3. Preparación previa

Ejecutar una vez en el equipo de presentación:

```bash
cd ~/practica_01_dl_jmfc
```

```bash
git pull origin main
```

```bash
source .venv/bin/activate
```

```bash
mkdir -p forecasting/data forecasting/artifacts forecasting/reports mlflow-data mlruns
```

```bash
export HOST_UID="$(id -u)"
```

```bash
export HOST_GID="$(id -g)"
```

Comprobar que existen el modelo final y el archivo de evaluación:

```bash
test -f forecasting/artifacts/best_model.pt && echo "Modelo final: OK"
```

```bash
test -f forecasting/reports/evaluation.json && echo "Evaluación: OK"
```

Si alguno no existe, preparar los artefactos con antelación:

```bash
dvc repro
```

Después verificar:

```bash
python -m forecasting.evaluate
```

## 4. Comandos exactos para el día de la presentación

### Terminal 1: iniciar el servidor

Abrir una terminal de WSL y ejecutar, uno por uno:

```bash
cd ~/practica_01_dl_jmfc
```

```bash
source .venv/bin/activate
```

```bash
export HOST_UID="$(id -u)"
```

```bash
export HOST_GID="$(id -g)"
```

```bash
docker compose -f docker-compose.forecasting.yml up -d --build forecasting-inference
```

```bash
docker compose -f docker-compose.forecasting.yml ps
```

```bash
curl http://127.0.0.1:5002/health
```

La respuesta debe indicar `status: ok`, modelo `lstm`, dispositivo `cpu` y
longitud de secuencia `24`.

### Terminal 2: ejecutar las pruebas

Abrir una segunda terminal y ejecutar:

```bash
cd ~/practica_01_dl_jmfc
```

```bash
source .venv/bin/activate
```

```bash
bash forecasting/tests/curl_tests.sh
```

La última línea debe ser:

```text
All inference curl tests passed.
```

### Mostrar una predicción manual

```bash
curl -X POST http://127.0.0.1:5002/predict -H 'Content-Type: application/json' -d '{"values":[10,11,12,13,14,15,16,17,18,19,20,19,18,17,16,15,14,13,12,11,10,9,10,11]}'
```

Mostrar los metadatos del modelo:

```bash
curl http://127.0.0.1:5002/model
```

Abrir la documentación interactiva:

```text
http://127.0.0.1:5002/docs
```

### Mostrar los resultados

```bash
cat forecasting/reports/evaluation.json
```

Los resultados esperados de la configuración mejorada son aproximadamente:

```text
LSTM: RMSE 0.7253 grados Celsius
RNN:  RMSE 0.7330 grados Celsius
```

### Mostrar MLflow, si se desea

En una tercera terminal:

```bash
cd ~/practica_01_dl_jmfc
```

```bash
source .venv/bin/activate
```

```bash
mlflow ui --backend-store-uri sqlite:///mlflow-data/mlflow.db --host 127.0.0.1 --port 5003
```

Abrir:

```text
http://127.0.0.1:5003
```

### Mostrar GitHub Actions

Abrir en el navegador:

```text
https://github.com/JoseMiguelFonsecaCordoba/practica_01_dl_jmfc/actions
```

Seleccionar el workflow `Forecasting CI` más reciente y mostrar los pasos
verdes: pruebas, compilación Python y pre-commit.

### Finalizar

Cuando termine la demostración:

```bash
docker compose -f docker-compose.forecasting.yml down
```

Si MLflow sigue activo, detenerlo en su terminal con:

```text
Ctrl+C
```

## 5. Qué se está haciendo de principio a fin

### 5.1 Dataset

Se utiliza Jena Climate, que contiene observaciones meteorológicas tomadas
aproximadamente cada diez minutos. El proyecto selecciona la columna
`T (degC)`, que representa la temperatura.

Como seis observaciones de diez minutos son aproximadamente una hora, se toma
una muestra cada seis registros. Así se reduce el tamaño de los datos y se
trabaja con una frecuencia horaria aproximada.

### 5.2 División temporal

Los datos se dividen en orden cronológico:

- 70% para entrenamiento.
- 15% para validación.
- 15% para prueba.

No se mezclan aleatoriamente las partes. En una serie temporal, usar datos
futuros para entrenar produciría una evaluación demasiado optimista.

### 5.3 Normalización

Se calcula la media y la desviación estándar únicamente con el conjunto de
entrenamiento:

```text
valor_normalizado = (valor - media_entrenamiento) / desviacion_entrenamiento
```

La misma media y desviación se utilizan para validación, prueba e inferencia.
Esto evita fuga de información y ayuda a que la red trabaje con valores
numéricamente estables.

### 5.4 Ventanas temporales

Cada muestra contiene 24 temperaturas consecutivas y la etiqueta es la
temperatura siguiente:

```text
[t1, t2, ..., t24] -> t25
```

Por tanto, el modelo recibe una secuencia de 24 horas y predice una hora
posterior. El endpoint rechaza cualquier secuencia que no tenga exactamente
24 valores.

### 5.5 Modelos

La RNN procesa la secuencia mediante una capa recurrente y usa el estado oculto
del último instante para producir una salida escalar.

La LSTM también procesa secuencias, pero utiliza una memoria y compuertas para
controlar qué información conserva, olvida o actualiza. Por eso suele manejar
mejor dependencias temporales largas.

Ambos modelos reciben los mismos datos, tienen una tarea idéntica y se evalúan
con la misma métrica. La comparación es así justa.

### 5.6 Entrenamiento

El entrenamiento utiliza:

- Función de pérdida MSE.
- Optimizador Adam.
- Batch de 64 muestras.
- Semilla aleatoria fija.
- 64 unidades ocultas en la configuración final.
- Tasa de aprendizaje `0.0005`.
- Hasta 40 épocas.
- Early stopping con paciencia de 5 épocas.

En cada época se calcula la pérdida de entrenamiento y validación. Se conserva
el estado del modelo que obtiene la menor pérdida de validación. Early stopping
detiene el entrenamiento si dejan de observarse mejoras.

### 5.7 MLflow

MLflow registra cada ejecución y guarda:

- Modelo utilizado.
- Número de épocas.
- Tamaño de la capa oculta.
- Tasa de aprendizaje.
- Pérdidas por época.
- Mejor pérdida de validación.
- Dispositivo usado.
- Modelo serializado.

El backend de seguimiento es SQLite y los artefactos se almacenan localmente.
Esto permite comparar RNN y LSTM sin depender de un servicio remoto.

### 5.8 DVC

DVC define tres etapas:

1. `prepare-data`: prepara o descarga el dataset.
2. `train`: ejecuta el entrenamiento dentro de Docker.
3. `select-best`: compara los modelos y copia el ganador como
   `forecasting/artifacts/best_model.pt`.

El archivo `dvc.lock` registra las versiones y hashes de las entradas y
salidas. Por eso `dvc repro` puede determinar qué etapas deben repetirse.

### 5.9 Selección del modelo

La selección se hace utilizando la menor pérdida de validación, no mirando
manualmente una predicción aislada. Después se evalúan ambos modelos en el
conjunto de prueba.

El resultado final fue:

| Modelo | MSE normalizado | RMSE |
|---|---:|---:|
| LSTM | 0.007023 | 0.7253 grados Celsius |
| RNN | 0.007172 | 0.7330 grados Celsius |

La LSTM se selecciona porque obtuvo menor RMSE en prueba.

### 5.10 FastAPI

El servidor carga `best_model.pt` al iniciar. El checkpoint contiene:

- Nombre del modelo.
- Pesos entrenados.
- Tamaño de la capa oculta.
- Longitud de secuencia.
- Media y desviación para desnormalizar.
- Pérdida de validación.
- Identificador de la ejecución MLflow.

Los endpoints son:

- `GET /health`: confirma que el servicio está activo.
- `GET /model`: muestra información del modelo.
- `POST /predict`: normaliza los 24 valores, ejecuta la red y devuelve la
  predicción en grados Celsius.

El servidor utiliza CPU en inferencia. La GPU se reserva para entrenamiento,
aunque el mismo código podría utilizar CUDA si se configura explícitamente.

### 5.11 Docker

Docker garantiza un entorno consistente con PyTorch y las dependencias del
proyecto. El servicio de entrenamiento y el servicio de inferencia están
separados:

- `forecasting-train`: prepara datos y entrena.
- `forecasting-inference`: sirve el modelo por HTTP.

Los directorios locales se montan como volúmenes para conservar datos,
checkpoints y reportes fuera del contenedor.

### 5.12 Validación

Las pruebas unitarias verifican las formas y salidas de RNN y LSTM.

El script `curl_tests.sh` comprueba:

1. Que `/health` responda correctamente.
2. Que una secuencia de 24 valores produzca una predicción.
3. Que una secuencia inválida produzca HTTP 422.

GitHub Actions repite automáticamente las pruebas unitarias, la compilación de
los módulos Python y los hooks de pre-commit.

## 6. Guion oral breve

Se puede explicar el proyecto así:

> El objetivo fue construir un pipeline reproducible para predecir la
> temperatura de la siguiente hora a partir de las 24 horas anteriores.
> Utilicé Jena Climate, tomé la temperatura y muestreé los datos cada seis
> registros para aproximar una frecuencia horaria. Dividí la serie en
> entrenamiento, validación y prueba respetando el orden temporal, y calculé
> la normalización únicamente con entrenamiento.
>
> Implementé una RNN y una LSTM en PyTorch. Las entrené con los mismos datos,
> registré los experimentos en MLflow y utilicé DVC para organizar las etapas
> de preparación, entrenamiento y selección. La configuración final usa 64
> unidades ocultas, learning rate 0.0005 y early stopping. La LSTM obtuvo un
> RMSE de 0.7253 grados Celsius, mejor que la RNN con 0.7330, así que la
> seleccioné como modelo final.
>
> Finalmente, guardé el checkpoint ganador y lo desplegué en FastAPI dentro de
> Docker. El endpoint recibe exactamente 24 valores, aplica la misma
> normalización, ejecuta la LSTM y desnormaliza el resultado. Las pruebas
> unitarias, las pruebas con curl, Ruff, pre-commit y GitHub Actions confirmaron
> que el sistema funciona.

## 7. Preguntas y respuestas posibles

### ¿Por qué utilizaste una serie temporal?

Porque el objetivo requiere predecir un valor futuro usando observaciones
anteriores. Una serie temporal permite demostrar ventanas, orden cronológico,
normalización y evaluación sin usar información futura.

### ¿Por qué Jena Climate?

Es un dataset público, conocido y suficientemente grande para entrenar modelos
recurrentes. También contiene observaciones frecuentes, lo que permite crear
ventanas temporales.

### ¿Por qué tomaste un registro cada seis observaciones?

El dataset tiene aproximadamente una observación cada diez minutos. Tomar cada
sexto registro aproxima una observación por hora y reduce el costo de
entrenamiento.

### ¿Por qué la ventana tiene 24 valores?

Porque representa aproximadamente las 24 horas anteriores. Es una ventana
interpretable para capturar un patrón diario y además coincide con el requisito
del endpoint.

### ¿Qué evita la fuga de información?

La media y la desviación estándar se calculan solo con entrenamiento. Además,
la división se realiza cronológicamente y no se mezclan observaciones futuras
con las pasadas.

### ¿Por qué comparaste RNN y LSTM?

Ambas son redes recurrentes adecuadas para secuencias, pero la LSTM incorpora
memoria y compuertas para manejar dependencias largas. Compararlas permite
medir si esa capacidad adicional mejora el forecasting en este dataset.

### ¿Qué significa RMSE?

Es la raíz del error cuadrático medio. Está expresado en las mismas unidades de
la variable original, grados Celsius. Menor RMSE significa una predicción más
cercana en promedio a la temperatura real.

### ¿Por qué seleccionaste la LSTM?

Porque obtuvo el menor error en el conjunto de prueba: aproximadamente
0.7253 grados Celsius frente a 0.7330 de la RNN.

### ¿Por qué el entrenamiento puede ser rápido?

La tarea predice un solo valor, utiliza una sola variable y la arquitectura es
pequeña. Además, el dataset se reduce a frecuencia horaria y se usa GPU cuando
está disponible. El tiempo de entrenamiento no es una medida directa de calidad.

### ¿Qué hace early stopping?

Detiene el entrenamiento cuando la pérdida de validación deja de mejorar
durante varias épocas. Conserva el mejor estado y reduce el riesgo de
sobreajuste y de entrenamiento innecesario.

### ¿Por qué usaste MLflow?

Para registrar parámetros, métricas, pérdidas y modelos de forma organizada.
Así la comparación entre ejecuciones es reproducible y no depende de recordar
manualmente los valores.

### ¿Por qué usaste DVC?

Para describir las etapas y dependencias del pipeline. DVC sabe qué salidas
dependen de qué archivos y permite reproducir la preparación, el entrenamiento
y la selección del modelo.

### ¿DVC está usando un almacenamiento remoto?

No. En esta versión se utiliza almacenamiento local porque la autenticación
con Google Drive presentó problemas en WSL. El pipeline DVC funciona
localmente; un remoto sería una mejora para compartir datos y checkpoints
entre equipos.

### ¿Por qué Docker?

Para fijar las dependencias y separar entrenamiento de inferencia. Así el
proyecto puede ejecutarse con un entorno consistente y los artefactos se
conservan mediante volúmenes.

### ¿Por qué entrenas con GPU pero sirves con CPU?

El entrenamiento requiere más operaciones y se beneficia de GPU. La inferencia
de una sola secuencia es pequeña, por lo que CPU es suficiente y simplifica el
despliegue.

### ¿Qué ocurre si envío menos de 24 valores?

FastAPI responde con HTTP 422 y el mensaje `Exactly 24 values are required`.
Esto evita que el modelo reciba una forma inesperada.

### ¿Cómo se convierte la predicción a grados Celsius?

La red predice en la escala normalizada. El servidor aplica la transformación
inversa:

```text
prediccion_celsius = prediccion_normalizada * desviacion + media
```

La media y la desviación están guardadas en el checkpoint.

### ¿Qué pruebas ejecutaste?

Dos pruebas unitarias para los modelos, tres pruebas funcionales con curl,
compilación de módulos Python, Ruff, formato Ruff y pre-commit. GitHub Actions
ejecuta automáticamente los checks principales.

### ¿Qué limitaciones tiene el proyecto?

Utiliza una sola variable de entrada y predice solo un paso futuro. No realiza
forecasting multihorizonte y DVC todavía no usa un remoto. Estas son extensiones
naturales para una versión posterior.

### ¿Qué mejorarías después?

Añadiría más variables meteorológicas, predicción de varias horas, validación
walk-forward, más ejecuciones con distintas semillas y un remoto DVC.

### ¿Por qué no hiciste un frontend?

No era necesario para cumplir el objetivo. FastAPI proporciona `/docs`, una
interfaz interactiva suficiente para demostrar el endpoint sin introducir
complejidad adicional ni distraer del pipeline de deep learning.

### ¿Por qué el endpoint usa `/predict` y no Triton?

El requisito era desplegar el mejor modelo sin Triton. FastAPI permite exponer
una API HTTP sencilla y controlar explícitamente la normalización, validación y
desnormalización.

## 8. Respuesta final si preguntan por el estado del proyecto

El proyecto está funcional y validado. La única preparación pendiente es
operativa: compilar el informe en Overleaf y conservar los artefactos locales
para la demostración. No es necesario añadir un frontend ni configurar un
remoto DVC para cumplir el alcance actual.
