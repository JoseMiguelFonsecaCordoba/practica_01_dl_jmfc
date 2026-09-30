import torch
from functools import cache

# Decorador de cache, el dipositivo se guarda y no se tiene que
# ejecutar otra vez
"""
@cache
def get_device():
    return torch.device("cpu")
"""

#como me estan pidiendo usar explicitamente cpu comentare esto

@cache
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    # Esto solo si tengo mac jeje
    # elif torch.bacends.mps.is_available():
    #     return torch.device('mps')
    else:
        return torch.device('cpu')
