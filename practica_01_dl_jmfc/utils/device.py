from functools import cache

import torch


@cache
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
