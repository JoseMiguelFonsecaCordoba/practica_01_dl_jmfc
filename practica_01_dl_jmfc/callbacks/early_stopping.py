import copy

# Paciencia: numero de epocas que estare esperando por una mejora
# min_delta: cuanto quiero que mi modelo mejore
class EarlyStopping:
    def __init__(self, patience: int=3, min_delta:float=1e-6) -> None:

        # Variables miembro de mi clase
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float('inf')
        self.counter = 0
        # Bandera: Si necesito detener o sigo entrenando
        self.early_stop = False
        # Aqui guardo los pesos de mi modelo cada que mejore
        self.best_model_state = None

    # Funcino miembro, recibo el modelo y la perdida
    # Evalua el modelo para determinada perdida
    def __call__(self, model, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            # No necesito incrementar la paciencia porque el modelo sigue funcionando
            self.counter = 0

            # Necesito hacer una copia profunda no solamente una asignacion
            self.best_model_state = copy.deepcopy(
                model.state_dict()
            )
        else:
            # Si es falso la perdida actual es peor que la mejor historica
            # incremento contador en 1
            self.counter += 1
            if self.counter >= self.patience:   
                self.early_stop = True

    # No se retorna nada porque es paso por referencia
    def restore_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state) 