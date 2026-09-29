import torch
import torch.nn as nn
from utils.device import get_device
from datasets.mnist import get_mnist_loaders
from models.mlp import MLP
from engine.trainer import evaluate, fit
from utils.plotting import plot_history
from callbacks.early_stopping import EarlyStopping
from models.cnn import CNN
import random
from models.factory import create_model


def main() -> None:
    ## Hiperparametros: Aquellos que yo modifico y cambian la forma de entrenamiento
    # 3 Formas diferentes de elegir hiperparametros
    # Busqueda en rejilla -> NO USAR
    # Busqueda aleatoria (inicio a fin) -> USAR A VECES
    # Busqueda bayesiana (Usando teorema de bayes optimizo parametros) -> LA MEJOR
    batch_size = 128
    learning_rate = 1e-3
    # Gradiente es la direccion de mayor cambio
    # El optimizador me dice hacia donde y el learning rate
    # me dice cuanto voy a avanzar hacia esa direccion
    # Este es un learning rate fijo durante todo el entrenamiento
    # El prolema es que cuando estamos cerca de ese minimo local, 
    # un learning rate grande puede afectar y nunca llegar al minimo global
    # Por eso necesitamos un scheduler, los mas usados son estos 3:
    # StepLR -> Cada N epocas reduzco el learning rate
    # ExponentialLF
    # ReduceROnPlateau -> En cuanto nuestra perdida deja de mejorar 
            # reduce el ratio de aprendizaje, conforme vamos llegando
            # al final de nuestro entrenamiento 


    # Un modelo normalmente se entrena por muchas epocas, si mi modelo es muy pequenio
    # empieza a funcionar peor, si es demasiado grande tambien empieza a funcionar peor
    # epochs = 1000
    # callback - early stopping
    # Por eso tengo algo que se llama callback: Pedazo de codigo que esta escuchando algo
    # va a estar escuchando mi perdida 
    # Definimos una paciencia, si en la epoca 80 la perdida sube y luego sube mas en las 
    # siguientes epocas, paro de entrenar porque ya no vale la pena seguir eentrenando,
    # cada epoca escucha mi epoca, tiene una paciencia y cuando la paciencia se acaba 
    # detiene el entrenamiento, es otra forma de regularizar, es otra forma de evitar
    # overfitting
    # Yo se que esta generalizndo porque los datasets de train != val, early stopping
    # me sirve porque si yo sigo entrenando hace overfitting, para entrenamiento y vuelvo
    # a cargar los pesos mejores de mi entrenamiento. 

    # En cada epoca desde la 1 hasta la n le tomo screenshot cada que mejora, hasta llegar a 
    # la eopoca en donde no mejora, si no mejora y se agota mi paciencia hago un ctrl+z a la ultima 
    # epoca donde mejoro lo mejor

    # Ya se puede guardar a disco duro
    epochs = 100

    # El numero de epocas que esperamos a que mejore depende del problema
    patience = 6
    min_delta = 1e-3
    # Cualquier numero hardcodeado/parametro cambia como mi modelo aprende
    weight_decay = 1e-4


    # Un mlp bien entrenado puede resolver la mayoria de problemas

    device = get_device()
    # Si yo ejecuto este programa me tiene que decir el device donde se este ejecutando
    # mps es el acelarador de los procesadores m de apple
    print(f"Training on {device}")


    train_loader, val_loader, test_loader = get_mnist_loaders(
        "data",
        batch_size=batch_size,
    )

    ######################### Model ###################################
    # model = MLP()
    # ------------------ Crear modelo CNN -------------------------------
    # model = CNN()
    # model = model.to(device)
    model = create_model("vgg11")
    model = model.to(device)
    ####################################################################


    ## Train 
    # Usamos entropia cruzada porque el problema es de clasificacion
    criterion = nn.CrossEntropyLoss()
    # Nos permite actulziar los pesos de la red
    # Graidente estocastico
    # Actualizacion de parametros
    # Estocastico algunas epocas actualiza algunas no
    optimizer = torch.optim.AdamW(
        # Le paso los parametros de mi modelo porque son los que voy a actualizar/optimizar
        model.parameters(), # Cambian los pesos de mi modelo y cambian a traves de mi entrenamiento
        lr=learning_rate,
        # Momenum: Agregamos un historial
        # Ayuda a suavizar mis gradientes, entreno mas rapido y mejor
        # momentum=0.9
        # Esto lo unico que hace es decirle al modelo que mantenga sus
        # parametros pequenios
        weight_decay=1e-4
    )

    sceduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # Trabajamo sobre el obtimizador
        optimizer,
        # Queremos minimizar 
        mode='min',
        # Cada que mi scheduler se active va a multiplicar el 
        # LR por 0.1
        factor=0.1,
        # Si en 2 epocas no mejora el LR se multiplica por el factor
        patience=2
    )

    early_stopping = EarlyStopping(
        patience = patience,
        min_delta = min_delta,
    )

    history = fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs,
        early_stopping,
        # sceduler
    )
    torch.save(
        # Guardando el estado interno del modelo
        model.state_dict(),
        "artifacts/best_model.tbh"
    )

    test_loss = evaluate(
        model, 
        test_loader,
        criterion,
        device
    )
    print(f"Test loss: {test_loss}")
    plot_history(history)

    # Dropout desativa mis neuronas, transfforma capa conecta a no conectada durante el 
    # entrenamiento, agrega ruido, mi modelo aprende mejor y no memoriza, GENERALIZA
    # Regularizacion


if __name__ == "__main__":
    main()

# data donde van mis modelos
# Dataset scripts para cargar los datos
# eniges scripts como entrenar
# models cargar los modelos
# utils dispositivo en donde voy a entrenar
# train.py defino hiperparametros, cargo dataset modelo y defino criterios de entrenamiento y entreno

# Siempre que yo tenga dropout mi validacion va a estar abajo
# Al visualizar una grafica de perdido vemos si necesitamos cambiar modelo, validacion, dataset
# Si mi validacion empieza a ir hacia arriba cuando ya convergio (Overfitting)
# Si desde un inicio empieza arriba y nunca baja (Underfitting)-> modelo muy pequenio para 
# la cantidad de datos que tengo

# O mi modelo es muy grande
# O no tengo regularizacion
# Data leakage
# Data drifting

# Siguiente clase: Regularizacion 
# Con aprendizaje profundo: ccms, llms, modelos difusos o autoencoders, levantar servidores de 
# inferencia con modelos ya entrenados
# 

# Tener un buen dataset
# Seleccionar un buen modelo para ese dataset
# El scheduler no sirvio de nada si los datos son feos'
# Estamos usando una ANN

# Proxima clase 
# Optimizadores
# Proxima de la proxima clase
# Regularizacion
# Proxima de la proxima de la proxima clase
# Redes convolucionales (CNN, Transformadores)

# Optimizadores
# SGD -> Todos los pesos se modifican por N
# Adam -> Lo unico que hace es como se calculan los pesos,
# no calcula el optimizador
# cambia como entrenamos 
# Adam -> Actualiza pesos de distinta manera
# ajuste fino de parametros, cada peso se va a estar ajustando
# de forma diferente a todos los demas
# Mi funcion de activacion decide que neurona queda prendida y cual apagada
# si yo uso Adam no generaliza
# AdamW -> Adam mejorado con Regularizacion para evitar que los parametros
# tiendan a infinito 
# SGD + MOmentum -> Siempre va a ser mejor que Adam

# FUNDAMENTOS
# Optimizador(SGD,AdamW,Adam): determina como atualizamos parametros
# lr: controla la magnitud de esas actualizaciones
# scheduler: modifica ese learning rate durante el entrenamiento

# - Cuando usamos SGD actualizamos todos los parametros al mismo tiempo de la misma forma
#       agregar momentum es agregar un historico para estabilizar nuestro entrenamiento para
#       que la historia pasada nos ayude al siguiente paso
#       para que el cambio de gradiente no sea tan agresivo
# - Adam actualiza cada peso de manera individual y tambien tiene el historico
#       es como si fuera SGD + momentum

# Todo valor que yo use para modificar como entrena el modelo, es un hiperparametro
# y si es un hiperparametro los puedo optimizar
# Estudio de hablacion: Buscar cuantas neuronas ocupa mi modelo para funcionar


# Deep Learning
# Un mlp es parte de machine learning, en ml necesitamos preprocesamiento
# hacemos feature engineering y en este modelo no sirve, nosotrs
# no estamos aprendiendo porque la mayoria de pixelles son negros y tan pocos 
# blancos, nunca aprende nada. Por eso en machine learning nosotros ocupamos 
# caracteristicas, por ejemplo hay algoritmos que hacen mas gordo mi trazado, 
# Otros que detectan lineas horizontales y verticales, en ml lo hacemos manual#
# Al hacerlo manual sesgamos al modelo, yo veo al 7 y digo es una linea vertical y
# otra horizontal, aqui entra deep learning

# Deep learning consta de 2 capas, 1 va a ser un ANN Normal pero esta se va a acoplar
# a un extractor de caracteristicas, aqui entran convoluciones, transformadores
# Lo que hace es extraer caracteristicas de forma automatica, no sesgamos a definir 
# las caracteristicas manualmente, detecta lo que el cree que es mejor 
# en base a la retropropagacion del error, empieza a extraer caracteristicas
# de forma autonoma, les dicen cajas negras pero si se sabe
# En lugar de meter la imagen cruda empeamos con un extractor de caracteristicas y 
# redes neuronales al final normales

# Modelos de vision, multimodales. 
# Puedo poner otro al otro lado y tengo un autoencoder
# genero ruido al eatorio al inicio y tengo uno difuso

# LA BASE ES EXTRACTOR DE CARACTERISTICAS
# Una convolucion es una forma matematica de unir dos seniales, tengo
# un kernel convolucional, tiene numeros y lo pongo arriba y lo multiplico y sumo
# el resultado lo pongo
# El kernel resalta las caracteristicas, hay un buen de kernels, pero si 
# decidimos a priori antes de empezara a entrenar cesgamos al modelo
# lo mejor es que el kernel se disenie solo el kernel se convierte en pesos
# El tamanio del kernel se puede cambiar y actualizar con la retropropagacion


# Clase de hoy era optimizadores septiembre 1 2026
# La siguiente acabamos redes neuronales con regularizacion
# Luego seran redes convolucionales 

# Como ya tendremos 2 modelos entrenados ann y cnn vamos a ver como deplegarlos
# Tenemos ONNX es un framework que se usa para servidor de inferencia como 
# VLLM -> Servidor para modelos de lenguaje
# Triton -> es el que usa en este proyecto dentro del contenedor de docker


# Con mlp tomamos los datos crudos y ycon esos datos crudos trata de construir 
# esos, no toda la imagen es informacion, son datos. En MNIST tenemos el numero 
# en medio y lo demas en negro, una cnn si lo sabe, puede resumir vectores tridimensionales 
# en una cnn no nos interesa aplanar esa imagen, la metemos en forma de matriz y 
# los metemos en kernels computacionales

# 1 0
# 0 1

# Tenemos k = 2, padding = 1 (Agregar 0 alrededor de la matriz), si tengo padding = 2(Agrego 2 ceros)
# Tengo Stride = 1, yo proyecto mi kernel sobre mi imagen (kernel 2x2) tengo

# En este tipo de redes siempre vamos a tener ReLU (Rectificadores lineales)
# Voy a constuir otro tensor poniendo la multiplicacion del kernel con la imagen\
# Stride: cuantas veces voy a brincar pixeles

# EL tamanio del kernel determina una atencion global, una cnn su atencion es local
# Una cnn tiene una camarita que se enfoca en una parte de la imagen
# Cada que yo hago backpropagation para cambiar errores los valores del kernel convolucional
# se convierten en mis pesos, puedo tener n kernels(En mi cnn puedo decir que quiero 16 kernels)
# Entonces tendria 28x28x16, ahora tengo un feature map, ya no es mi imagen original, ahora 
# se transformo en caracteristicas gracias al kernel, ahora detecta mas todo lo que sirva para clasificar

# El feature map si yo uso padding 0 me tuumba 1 o 2 pixeles

# Problema de las redes convolucionales, si yo tengo 16 kernels en profundidad lo hago 16 veces
# consumo mas memoria y hago un procesamiento mas pesado

# Una red convolucional va a usar mucha mas memoria, por eso una cnn o capa convolucional
# va de la mano de una capa de pooling, es el mismo proceso pero sin un kernel convolucional
# si yo escojo una capa de pooling de 2x2, tomo el kernel pero vacio, hay max y avg. Si 
# es 2x2 tomo 4 pixeles de la imagen para sobreponer sobre la imagen
# mi convolucional resalta caracteristicas, el pooling mantiene esas caracteristicas pero 
# reduciendo dimensionalidad, ahora no tendre 28x28x16, tendre 14x14x16
# Capa convolucional 
# Pooling mantiene caracteristicas que el cnn encontro

# Ahora tenemos una parte que extrajo caracteristicas (Cnn), Otro para reducir dimensionalidad(Pooling)
# y finalmente tengo la red normaal para segmentar, clasificar detectar. Un mlp se conecta al final de una CNN
# que por si sola solo resalta caracteristicas/patrones

# El feature map o puedo conectar tambien a un knn, a un arbol, un xgboost, generalmente es un mlp pero nosotros podemos
# hacer lo que queramos

# TRES VARIANTES DEL APRENDIZAJE
# Aprendizaje supervisado: Un humano preclasifico el dataset. Yo tengo la verdad y lo que mi modelo predice, saco una metrica
# del error y con base a esa etiqueta yo actulizo hiperparametros
# Aprendizaje no supervisado: No tengo dataset etiquetado
# Aprendizaje por refuerzo: Se premia y castiga al modelo, le damos puntuacion positiva si hizzo algo bien, y una negativa
# si hizo algo mal, no necesariamente etiquetado

# Existe la operacion inversa a la convolucion
# En este punto lo que tenemos es un modelo en los que tenemos imagenes, si tuvierammos imagenes medicas son de mayor dimensionalidad
# es importante la reduccion de la dimensionalidad, terminamos con 32x7x7, aqui conectamos a nuestra red neuronal, esto 
# es un modelo de aprendizaje profundo, una parte que es un extractor, y otra que es un clasificador/detetor/segmentaor/regresor
# ESTO ES APRENDIZAJE PROFUNDO
# Del lado iziquierdo tengo cnn, puedo tener un transformador, recurrentes, temporales a corto o largo plazo, lo que sea
# A la derecha knn, xgboost, random forest, siemrpre y guando tengamos un modelo extractor clasificador se le llama aprendizaje
# profundo
# Supongamos que en lugar de poner como clasificador un mlp, le tomo foto al modelo de deep learning y lo invierto, entonces
# ahora tengo un modelo que se llama autoencoder, el cual comprime a un espacio latente y descomprime tratando descubrir la 
# imagen original, sirve para generar datos artificiales, o en caso de que yo sepa que la reconstruccion tiene un errorr
# pequenio, entonces yo se que la reduccion del espacio latente es la mejor representacion del espacio latente. La puedo 
# tener incluso hasta para tener un almacen de fotos 

# Tenemos que saber convolucionales y transformadores, con que sepamos explicar porque la atencion funciona

# En deep learning entre mas grande sea el modelo mejor 

# trasnfer learning, carga el modelo original, 
# todas las capas convolucionales ya no las necesito para entrenar,
# solo necesito la parte de clasificacion
# congelo la gran mayoria de mi modelo y solo entreno la parte de clasificacion, 
# es un modelo que ya esta preentrenado, no tengo que entrenar todo el modelo

