import torch

def train_one_epoch(
        model, 
        dataloader,
        # Funcion de perdida
        criterion,
        optimizer,
        device
):
    # Cuando vamos a entrenar, tenemos que poner el modelo en modo entrenamiento
    # Si no no entrena
    model.train()
    total_loss = 0
    # total_correct = 0
    total_samples = 0
    for inputs, labels in dataloader:
        # Cuando leemos un dataset esta en la ram de la cpu
        # trasladamos de la ram a la vram para mandarlo al dispositivo
        # donde voy a entrenar
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Regreso los logits 
        outputs = model(inputs)
        # Actualizo mi perdida con mi funcion de perdida
        loss = criterion(outputs, labels)

        # En cada batch actualizo parametros 
        # Cada que yo veo 1000 elementos de mi dataset actualizo parametros
        # Descenso del gradiente normal es malo porque tengo mucha 
        # inestabilidad al entrenar
        # En un paso de optimizacion con respecto al anterior
        # mi cambio de gradiente puede ser muy brusco
        # Como ya calcule el error lo retropropago
        optimizer.zero_grad()
        # Calcula los gradientes en base a mmi funcion de perdida
        loss.backward()
        # Una vez que ya tengo los gradientes calculados actualizo
        # mis hiperparametros
        # Hago un salto de optimizador
        optimizer.step()

        # Perdida total con formato que yo tengo de entradas
        total_loss += loss.item() * inputs.size(0)

        total_samples += labels.size(0)
    avg_loss = total_loss/total_samples
    return avg_loss

# Al evaluar ya no necesito un optimizador
# porque no estoy entrenando, tengo que poner mi modelo
# en modo evaluacion
def evaluate(
        model, 
        dataloader,
        # Funcion de perdida
        criterion,
        # optimizer,
        device
):
    # Cuando vamos a entrenar, tenemos que poner el modelo en modo entrenamiento
    # Si no no entrena
    model.eval()
    total_loss = 0
    # total_correct = 0
    total_samples = 0
    # Esta solo es un paso hacia adelante calculo la perdida y listo
    with torch.no_grad():
        for inputs, labels in dataloader:
            # Cuando leemos un dataset esta en la ram de la cpu
            # trasladamos de la ram a la vram para mandarlo al dispositivo
            # donde voy a entrenar
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Regreso los logits 
            outputs = model(inputs)
            # Actualizo mi perdida con mi funcion de perdida
            loss = criterion(outputs, labels)

            # Ya no tengo que hacer la retropropagacion

            # Perdida total con formato que yo tengo de entradas
            total_loss += loss.item() * inputs.size(0)
            
            total_samples += labels.size(0)
    avg_loss = total_loss/total_samples
    return avg_loss

def fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        # Numero de epocas por las que voy a entrenar
        epochs,
        early_stopping=None,
        scheduler=None
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device
        )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            # 2e porque el numero puede ser muy pequenio
            f"Learning Rate: {current_lr:.2e}"
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        if early_stopping is not None:
            early_stopping(model, val_loss)
            # Si mi bandera es verdadera detetngo el entrenamiento
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    if early_stopping is not None:
        early_stopping.restore_best_model(model)

    return history

