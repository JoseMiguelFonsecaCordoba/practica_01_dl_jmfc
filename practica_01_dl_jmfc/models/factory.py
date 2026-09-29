from models.cnn import CNN
from models.mlp import MLP
from models.vgg import VGG11

def create_model(
        model_name,
        num_clases=10,
        pretrained=True,
):
    if model_name == "cnn":
        return CNN()
    if model_name == "mlp":
        return MLP()
    if model_name == "vgg11":
        return VGG11(num_classes=num_clases, pretrained=pretrained)

    raise ValueError("Unknown model")

    