import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # ----------------------------------------------------------
            # BLOQUE CONVOLUCIONAL 
            # CAPAS CONVOLUCIONALES: Extraen caracteristicas espaciales de mi
            # dataset (2D)
            # Una capa convolucional en 3d agrega la variable del tiempo
            # puede procesar video
            # Numero dimensiones (blanco negro 1, si fuera rgb serian 3)
            nn.Conv2d(in_channels=1, # Un canal de entrada porque es en blanco y negro
                      out_channels=16, # Va a trabajar con 16 kernels convolucionales
                      kernel_size=2,
                      stride=1, # Tamanio del paso
                      padding=1), # Numero de 0 alrededor 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(in_channels=16, # Por cada kernel convolucional se transforma en profundidad
                      out_channels=32,
                      kernel_size=3,
                      stride=1, # Tamanio del paso
                      padding=1), # Numero de 0 alrededor, tendrems 32 * 7 * 7
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #
            # Feature maps: Que pixeles estan activos y cuales no
            # Se usan para agregar explicabilidad (sharp, line) para ver
            # porque el modelo tomo la decision que tomo, es hacer el paso contrario
            # retroceder para ver lo que llamo la atencion en el modelo
            # Este Dropout funciona diferente, en el mlp desactiva neuronas, pero en 
            # una capa convolucional no hay neuronas, aqui desactiva un feature map 
            # y desactiva su porcentaje, hace que las caracteristicas entre mapas
            # de caracteristicas no se especializen en algo en especifico, sino 
            # que se especialce en generalizar
            # Pareciera que aprende mejor porque el training loc sube
            nn.Dropout2d(0.1),
        )

        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Aqui si me interesa la salida, 7*7 (tamanio del vector resultante) y como al final vamos a tener 
            # 32 filtros ponemos 32, tendremos informacion (curvas, textura, etc), lo metemos a una capa 
            # totalmente conectada de 128 neruonas, 
            nn.Linear(32*7*7, 128),
            nn.ReLU(),
            # Regularizacion: Apago el 50% de mis neuronas durante entrenamiento,
            # su finalidad es evitar neuronas especializadas
            # Esto hace que nuestas capas durante entrenamiento sean parcialmente 
            # conectadas
            # Va siempre despues de la funcion de activacion
            # SIEMPRE HAGO DROPOUT EN MI CAPA ANTERIOR A MI CAPA FINAL
            nn.Dropout(0.5),
            # Aqui no puedo apagar neuronas porque cada neurona es una clase
            nn.Linear(128, 10),
        )

    # En el paso adelante extraemos caracteristicas y despues de pasa por el mlp
    def forward(self, x):
        # Lo que entre a mi modelo primero lo paso a mi extractor de carracteristicas (cnn y pooling)
        x = self.features(x)
        # Despues lo paso al clasificador (mlp)
        x = self.classifier(x)
        return x # Loggits -> Salida cruda de nuestra capa