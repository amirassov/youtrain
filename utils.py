import random
import torch
import numpy as np
import yaml


def set_global_seeds(i: int):
    torch.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    random.seed(i)
    np.random.seed(i)


def get_config(path: str) -> dict:
    with open(path, 'r') as stream:
        config = yaml.load(stream)
    return config
