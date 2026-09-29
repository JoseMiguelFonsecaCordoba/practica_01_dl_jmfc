from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import torch

# Descarga el dataset, lo convierte a tensores y despues crea un dataloader
# Tecnica de regularizacion es el early stoping, cuando ya no mejora
# paro el entrenamiento y cargo los parametros 
# Tengo que dejar 20% para validar nunca debe haber sido usado para entrenar
# El objetivo es que generalice no que memorice
# para tener evaluaciones que nuunca vio y pueda generalizar y aprender bien
def get_mnist_loaders(data_dir, batch_size=64, val_split=0.2,):
    # Variable de transformacion(transforma algo a tensor)
    # Matriz 10 x 10 si agregamos capaz en profundidad se convierte en un tensor
    # Una imagen es un tensor porque tiene 3 dimensiones de profundidad

    # DATA AUGMENTATION: Se usa cuando tengo datos desbalanceados o cuando 
    # quiero normalizar
    # Las transformaciones dependen directamente del dataset
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1,0.1),
        ),
        transforms.ToTensor(),
        # MNIST, mean 0.1307, std 0.3081 (TRANSFORMACION Z)
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # En evaluacion solo convierto a tensor y normalizo, no necesito mas    
    eval_transform = transforms.Compose([
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
        # MNIST, mean 0.1307, std 0.3081 (TRANSFORMACION Z)
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform = transforms.ToTensor()
    full_train_dataset = datasets.MNIST(
        data_dir,
        # Entrenamiento
        train=True,
        # Si lo queremos descargar
        download=True,
        # AL final le pasamos la tranformacion que creamos para que convierta lo que descargamos a tensor
        transform=train_transform,
    )

    test_dataset = datasets.MNIST(
        data_dir, 
        train=False,
        # Descargamos la parte de validacion en lugar de entrenamiento
        download=True,
        transform=eval_transform,
    )

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        # Siempre inicializar una semilla 
        generator = torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # Este jamas lo tiene que ver el entrenamiento
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, val_loader, test_loader


# AJUSTE DE HIPERPARAMETROS
# LA MEJOR: OPTIMIZACION BAYESIANA (LA MAS DIFICIL), SOLO DEFINIMOS 
# PARAMETOS INICIALES, DESPUES CON LA BUSQUEDA BAYESIANA NOS DARA 
# LOS MEJORES PARAMETROS PARA NUESTRO ENTRENAMIENTO

# LO VAMOS A HACER NOSOTROS COMO TAREA WUE
# BUSQUEN HIPERPARAMETROS, EL PROFE DICE QUE CON BUSQUEDA ALEATORIA ES SUFICIENTE

# BUSQUEDA ALEATORIA: SUFICIENTE PARA ENCONTRAR HIPERPARAMETROS
# BUSQUEDA EN REJILLA


# JOHN SAVILL'S TECHNICAL TRAINING


# MODELO LOCAL: BUENO PARA GOBERNANZA DE DATOS
# MODELO FRONTERA