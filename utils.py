import random
import torch
import numpy as np
import yaml
from .parallel import DataParallelCriterion, DataParallelModel
from torch.nn import DataParallel


def set_global_seeds(i: int):
    torch.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    random.seed(i)
    np.random.seed(i)


def get_config(path: str) -> dict:
    with open(path, 'r') as stream:
        config = yaml.load(stream)
    return config


def batch2device(data, device):
    return {k: v if not hasattr(v, "to") else v.to(device) for k, v in data.items()}


def model_parallel(model, mode):
    if mode == 'pytorch':
        model = DataParallel(model)
    elif mode == 'criterion':
        model = DataParallelModel(model)
    return model


def criterion_parallel(criterion, mode):
    if mode == 'criterion':
        criterion = DataParallelCriterion(criterion)
    return criterion
