import pydoc
import torch
import os
from glob import glob
import numpy as np


class Factory:
    def __init__(self, params: dict, **kwargs):
        self.params = params
        self.kwargs = kwargs

    def make_model(self) -> torch.nn.Module:
        model_name = self.params['model']
        model = pydoc.locate(model_name)(**self.params['model_params'])
        if 'weights' not in self.params or self.params['weights'] is None:
            return model
        elif isinstance(self.params['weights'], str):
            model.load_state_dict(torch.load(self.params['weights'])['state_dict'], strict=False)
            return model
        else:
            raise ValueError("type of weights should be None or str")

    @staticmethod
    def make_optimizer(model, stage) -> torch.optim.Optimizer:
        for p in model.parameters():
            p.requires_grad = True
        if 'freeze_features' in stage and stage['freeze_features']:
            for p in model.module.features.parameters():
                p.requires_grad = False
        return getattr(torch.optim, stage['optimizer'])(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            **stage['optimizer_params'])

    @staticmethod
    def make_scheduler(optimizer, stage):
        return getattr(torch.optim.lr_scheduler, stage['scheduler'])(
            optimizer=optimizer,
            **stage['scheduler_params'])

    def make_loss(self) -> torch.nn.Module:
        return pydoc.locate(self.params['loss'])(
            **self.params['loss_params'])

    def make_metrics(self) -> dict:
        return {metric: pydoc.locate(metric)(**params) for metric, params in self.params['metrics'].items()}


class DataFactory:
    def __init__(self, params: dict, paths: dict):
        self.paths = paths
        self.params = params

    def make_train_loader(self):
        raise NotImplementedError

    def make_val_loader(self, **kwargs):
        raise NotImplementedError


class CheckpointLoader:
    def __init__(self, params: dict, paths: dict):
        self.paths = paths
        self.params = params

    def get_best_checkpoint(self):
        all_checkpoints = self.all_checkpoints
        if len(all_checkpoints) == 0:
            raise RuntimeError('No checkpoints found')

        best_checkpoint = self._chosen_best_checkpoint(all_checkpoints)
        print('Best_checkpoint: ', best_checkpoint)
        return best_checkpoint

    @property
    def all_checkpoints(self):
        print(os.path.join(self.weights_dir, '**', '*.pt'))
        checkpoints = glob(os.path.join(self.weights_dir, '**', '*.pt'), recursive=True)
        return checkpoints

    @staticmethod
    def _get_checkpoint_metric(checkpoint):
        return float(checkpoint.split('_')[-1].split('.pt')[0])

    @property
    def weights_dir(self):
        return os.path.join(self.paths['path'], self.paths['weights'], self.params['name'])

    def _chosen_best_checkpoint(self, checkpoints):
        checkpoints = np.array(checkpoints)
        metrics = [self._get_checkpoint_metric(ch) for ch in checkpoints]
        best_checkpoint = checkpoints[np.argmax(metrics)]
        return best_checkpoint
