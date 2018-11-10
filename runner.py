from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

tqdm.monitor_interval = 0


class Metrics:
    def __init__(self, functions):
        self.functions = functions
        self.best_score = float('inf')
        self.best_epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}


class Runner:
    def __init__(self, factory, callbacks, stages, device):
        self.stages = stages
        self.factory = factory
        self.device = device
        self.model = self.factory.make_model()
        self.model = nn.DataParallel(self.model).to(device)
        self.loss = self.factory.make_loss().to(device)
        self.metrics = Metrics(self.factory.make_metrics())

        self.current_stage = None
        self.global_epoch = 0
        self.optimizer = None
        self.scheduler = None

        self.callbacks = callbacks
        self.callbacks.set_trainer(self)

    def fit(self, data_factory):
        self.callbacks.on_train_begin()
        for stage in self.stages:
            self.current_stage = stage

            train_loader = data_factory.make_loader(stage, is_train=True)
            val_loader = data_factory.make_loader(stage, is_train=False)

            self.optimizer = self.factory.make_optimizer(self.model, stage)
            self.scheduler = self.factory.make_scheduler(self.optimizer, stage)

            self.callbacks.on_stage_begin()
            self._run_one_stage(train_loader, val_loader)
            self.callbacks.on_stage_end()

        self.callbacks.on_train_end()

    def _run_one_stage(self, train_loader, val_loader=None):
        for epoch in range(self.current_stage['epochs']):
            self.callbacks.on_epoch_begin(self.global_epoch)

            self.model.train()
            self.metrics.train_metrics = self._run_one_epoch(epoch, train_loader, is_train=True)

            self.model.eval()
            self.metrics.val_metrics = self._run_one_epoch(epoch, val_loader, is_train=False)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(self.metrics.val_metrics['loss'], epoch)
            else:
                self.scheduler.step(epoch)

            self.callbacks.on_epoch_end(self.global_epoch)
            self.global_epoch += 1

    def _run_one_epoch(self, epoch, loader, is_train=True):
        epoch_report = defaultdict(float)

        if is_train:
            progress_bar = tqdm(
                enumerate(loader), total=self.factory.params['steps_per_epoch'],
                desc=f"Epoch {epoch} training...", ncols=0)
        else:
            progress_bar = tqdm(
                enumerate(loader), total=len(loader),
                desc=f"Epoch {epoch} validating...", ncols=0)

        metrics = {}
        with torch.set_grad_enabled(is_train):
            for i, data in progress_bar:
                self.callbacks.on_batch_begin(i)
                step_report = self._make_step(data, is_train)
                for key, value in step_report.items():
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    epoch_report[key] += value

                metrics = {k: v / (i + 1) for k, v in epoch_report.items()}
                progress_bar.set_postfix(**{k: f"{v:.5f}" for k, v in metrics.items()})
                self.callbacks.on_batch_end(i, step_report=step_report, is_train=is_train)
        return metrics

    def _make_step(self, data, is_train):
        report = {}
        data = self.batch2device(data)
        images = data['image']
        labels = data['mask']

        if is_train:
            self.optimizer.zero_grad()

        predictions = self.model(images)
        loss = self.loss(predictions, labels)
        report['loss'] = loss.data

        for metric, f in self.metrics.functions.items():
            report[metric] = f(predictions, labels)

        if is_train:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            report['grad'] = grad_norm
            self.optimizer.step()

        return report

    def batch2device(self, data):
        return {k: v.to(self.device) for k, v in data.items()}
