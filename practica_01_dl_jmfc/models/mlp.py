import torch.nn as nn

# Herencia
# Usando herencia podemos desacoplar el funcionamiento de nuestro 
# software
# usando factory, tenemos un entrenamiento e inferencia desacoplada
class MLP(nn.Module):
    def __init__(self):
        # Constructor y mando llamar al constructor de la clase padre
        super(MLP, self).__init__()
        # Vertical de capas en orden, lo primero que le mandemos a nuestro
        # secuencial e lo primero que va a hacer la red
        self.network = nn.Sequential(
            nn.Flatten(),
            # Capa lineal 28*28 caracteristiacas de entrada
            # 128 Caracteristicas de salida
            # Dentro de la neurona tenemos una sumtoria que es el producto punto
            # El numero de neuronas no depende de la capa anterior
            # Si mi modelo es muy grande aprende, si es muy justo subaprende
            # 
            nn.Linear(28 * 28, 128),
            # Funcion de activacion para que mi capa anterior tenga mi funcion de activacion,
            # los modelos profundos entre mas capas tengamos necesitammos una propagacion
            nn.ReLU(),
            # Regularizacion para volver a entrenar con un dropout de 5
            # nn.Dropout(0.5)
            # Capa linear la salida de 128 de arriba entra aqui y finalmente pongo 10 salidas
            # porque tengo 10 clases
            nn.Linear(128, 10), # logits salida cruda de una red neuronal resultado de mi 
            # producto punto todavia sin activarlo 
        )

    # Paso hacia adelante   
    def forward(self, x):
        # Regreso los logits de mi modelo
        return self.network(x)


# Capa totalmente conectada
# Capa parcialmente conectada
# Si yo tengo 10 neuronas de entrada, y despues de esa capa tengo .5(50%)
# cuando hago propagacion hacia adelante en modo entrenamiento el dropout
# tumba el 50% de las neuronas de forma aleatoria, paso a tener una capa
# parcialmente conectada, cuando pasa a modo evaluacion pasa a ser una capa 
# totalmente conectada, hace mas dificil etrenar, hace que el dataset tenga 
# ruido y hace mas dificil el entrenamiento pero al validar no. 
# Esto me ayuda a que generalice mejor, de esta forma evito que memorice el dataset, 
# en cada epoca forzo al modelo. Cunado yo sobreausto una neurona se queda clavada con 
# lo que ya sabe, si yo la desacativo otras neuronas aprendan lo que esa neurona
# ya habia aprendido. 
# Dropout: Apaga neuronas de forma aleatoria para mejorar la generalizacion
