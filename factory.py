import pydoc
import torch


class Factory:
    def __init__(self, params, **kwargs):
        self.params = params
        self.kwargs = kwargs

    def make_model(self):
        model_name = self.params['model']
        model = pydoc.locate(model_name)(**self.params['model_params'])
        if 'weights' not in self.params or self.params['weights'] is None:
            return model
        elif isinstance(self.params['weights'], str):
            model.load_state_dict(torch.load(self.params['weights'])['state_dict'])
            return model
        else:
            raise ValueError("type of weights should be None or str")

    @staticmethod
    def make_optimizer(model, stage):
        for p in model.parameters():
            p.requires_grad = True
        if 'freeze_encoder' in stage and stage['freeze_encoder']:
            for p in model.module.encoder.parameters():
                p.requires_grad = False
        return getattr(torch.optim, stage['optimizer'])(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            **stage['optimizer_params'])

    @staticmethod
    def make_scheduler(optimizer, stage):
        return getattr(torch.optim.lr_scheduler, stage['scheduler'])(
            optimizer=optimizer,
            **stage['scheduler_params'])

    def make_loss(self):
        return pydoc.locate(self.params['loss'])(
            **self.params['loss_params'])

    def make_metrics(self):
        return {metric: pydoc.locate(metric)() for metric in self.params['metrics']}


class DataFactory:
    def __init__(self, params, paths, **kwargs):
        self.paths = paths
        self.params = params
        self.kwargs = kwargs

    def make_loader(self, stage, **kwargs):
        raise NotImplementedError
