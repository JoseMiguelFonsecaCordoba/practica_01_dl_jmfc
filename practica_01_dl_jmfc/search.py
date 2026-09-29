import random
import torch
import torch.nn as nn
from utils.device import get_device
from datasets.mnist import get_mnist_loaders
from models.mlp import MLP
from engine.trainer import evaluate, fit
from utils.plotting import plot_history
from callbacks.early_stopping import EarlyStopping
from models.cnn import CNN

def main() -> None:
    #PARÁMETROS FIJOS
    epochs = 100
    patience = 6
    min_delta = 1e-3

    #CONFIGURACIÓN DE LA BÚSQUEDA ALEATORIA 
    search_space = {
        "learning_rate": [1e-3, 5e-4, 1e-4, 5e-5],
        "weight_decay": [1e-3, 1e-4, 1e-5, 0.0],
        "batch_size": [64, 128, 256]
    }
    
    num_trials = 5 # Cuántas combinaciones al azar vamos a probar
    best_loss = float('inf')
    best_params = {}

    device = get_device()
    print(f"Training on {device}")

    # INICIO DEL CICLO DE BÚSQUEDA
    for trial in range(num_trials):
        # Seleccionamos hiperparámetros aleatorios para esta prueba
        learning_rate = random.choice(search_space["learning_rate"])
        weight_decay = random.choice(search_space["weight_decay"])
        batch_size = random.choice(search_space["batch_size"])

        print(f"\n{'='*40}")
        print(f"--- Prueba {trial + 1}/{num_trials} ---")
        print(f"Params: LR={learning_rate}, Weight Decay={weight_decay}, Batch Size={batch_size}")
        print(f"{'='*40}")

        # Pasamos el batch size seleccionado aleatoriamente
        train_loader, val_loader, test_loader = get_mnist_loaders(
            "data",
            batch_size=batch_size,
        )

        ## Model 
        # ------------------ Crear modelo CNN -------------------------------
        model = CNN()
        model = model.to(device)

        ## Train 
        criterion = nn.CrossEntropyLoss()
        
        # Pasamos el learning_rate y el weight_decay seleccionados
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Corregido el nombre de la variable de sceduler a scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=2
        )

        early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
        )

        # Agregamos el scheduler al entrenamiento para que cumpla su función
        history = fit(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs,
            early_stopping,
            scheduler
        )

        # Evaluamos el modelo en el conjunto de prueba
        test_loss = evaluate(
            model, 
            test_loader,
            criterion,
            device
        )
        print(f"Test loss en esta prueba: {test_loss:.4f}")

        # plot_history(history) # Lo mantenemos comentado para no pausar el ciclo 

        #GUARDAR EL MEJOR MODELO 
        if test_loss < best_loss:
            best_loss = test_loss
            best_params = {
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'batch_size': batch_size
            }
            # Guardamos con un nombre distinto para saber que viene de la búsqueda
            torch.save(
                model.state_dict(),
                "artifacts/best_model_search.tbh"
            )
            print("¡Nuevo mejor modelo guardado!")

    #RESULTADOS FINALES
    print(f"\n{'*'*40}")
    print("Búsqueda aleatoria finalizada.")
    print(f"Mejores parámetros encontrados: {best_params}")
    print(f"Mejor Test Loss global: {best_loss:.4f}")
    print(f"{'*'*40}")

if __name__ == "__main__":
    main()