from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.fedPDA.staralign import staralign_train
from src.ml_training.evaluate import evaluate_model
from src.ml_training.local_supervised_learning import local_train, compute_average_learning_direction
from src.utils.network_utils import estimate_bn_stats
from src.utils import argumentlib
from src.utils.argumentlib import args, config
import logging


class Client:
    def __init__(self, model: torch.nn.Module, optimizer_class: type, optimizer_params: dict, dataset_length: int,
                 name: str, device, train_dataloader, val_dataloader, test_dataloader,
                 server, criterion, scheduler_class: type = None, scheduler_params: dict = None):
        self._model = model
        self._optimizer = optimizer_class(self._model.parameters(), **optimizer_params)
        self._dataset_length = dataset_length
        self._scheduler = None
        self._name = name
        self._device = device
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

        if scheduler_class is not None:
            self._scheduler = scheduler_class(self._optimizer, **scheduler_params)

        self._server = server
        self._criterion = criterion

        self.best_selection_metric = 0.0

        # Move model to device
        self._model.to(device)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def optimizer(self) -> optim.Optimizer:
        return self._optimizer

    @property
    def dataset_length(self) -> int:
        return self._dataset_length

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def model_weights(self) -> dict:
        return self._model.state_dict()

    @property
    def name(self) -> str:
        return self._name

    def get_dataloader(self, train_test_val: str):
        if train_test_val == 'train':
            return self._train_dataloader
        elif train_test_val == 'val':
            return self._val_dataloader
        elif train_test_val == 'test':
            return self._test_dataloader
        else:
            raise ValueError(
                f'Invalid input: train_test_val must be either train, val or test but was {train_test_val}.')

    def update_model(self, new_model, strict=True) -> None:
        if isinstance(new_model, nn.Module):
            self._model.load_state_dict(new_model.state_dict(), strict=strict)
        elif isinstance(new_model, dict):
            self._model.load_state_dict(new_model, strict=strict)
        else:
            raise ValueError('Invalid input: new_model must be an instance of nn.Module or a state dictionary.')

    def estimate_bn_stats(self, train_test_val) -> None:
        estimate_bn_stats(model=self._model, dataloader=self.get_dataloader(train_test_val=train_test_val),
                          device=self._device)

    def train(self, method: str = 'default'):
        logging.info(f'Begin local training on client: {self.name}')
        if method == 'default':
            loss = local_train(model=self._model, optimizer=self._optimizer, dataloader=self.get_dataloader('train'),
                               device=self._device, loss_fun=self._criterion)
        elif method == 'staralign':
            # Initialize global model template with current client model
            self._server.global_model_template.load_state_dict(self._model.state_dict())
            avg_gradients = None
            # Compute the average gradient direction
            if self.name != self._server.target_client_name:
                staralign_optimizer = optim.SGD(self.model.parameters(), lr=args.lr)

                avg_gradients_loss, avg_gradients = compute_average_learning_direction(local_train)(model=self.model,
                                                                                                    optimizer=staralign_optimizer,
                                                                                                    dataloader=self.get_dataloader('train'),
                                                                                                    device=self._device,
                                                                                                    loss_fun=self._criterion)
                logging.info(f'Client {self.name} - Avg. gradient loss: {avg_gradients_loss:.2f}' + '\n' + '-' * 50)

            # Send to target client and execute there
            loss, new_weights = staralign_train(model=self._server.global_model_template,
                                                dataloader=self._server.target_client.get_dataloader('train'),
                                                device=self._device, average_grads=avg_gradients,
                                                beta=args.beta, loss_fun=self._criterion)
            self.update_model(new_weights)
        else:
            raise ValueError(f"Unsupported training method: {method}")

        logging.info(f'Client {self.name} - Training loss: {loss:.2}' + '\n' + '-' * 50)
        return loss

    def evaluate(self, train_test_val, num_classes, step, wandb_metrics, save_best_model=True, save_latest=False):
        logging.info('-' * 50 + '\n' + f'Evaluate client: {self.name} on step {step}')
        loss, correct, total, weighted_acc = evaluate_model(model=self.model,
                                                            dataloader=self.get_dataloader(train_test_val),
                                                            device=self._device, num_classes=num_classes,
                                                            loss_fun=self._criterion)

        if save_best_model:
            if config['evaluation_metric']['metric_name'] == 'accuracy':
                metric = correct / total
            elif config['evaluation_metric']['metric_name'] == 'weighted_accuracy':
                metric = weighted_acc
            else:
                raise ValueError(f"Unsupported evaluation metric: {config['evaluation_metric']['metric']}")
            if metric > self.best_selection_metric:
                self.best_selection_metric = metric
                self.save_model(which_model=config['evaluation_metric']['metric_name'], step=step)
        if save_latest:
            self.save_model(which_model='latest', step=step)

        # Log to console and wandb
        logging.info(f'Client {self.name} - {train_test_val} loss: {loss:.2f}' + '\n' + '-' * 50)
        if args.wandb_projectname is not None:
            self.update_wandb_metrics(wandb_metrics=wandb_metrics, train_val_test=train_test_val, loss=loss,
                                      accuracy=correct / total, weighted_accuracy=weighted_acc)

        return loss, correct, total, weighted_acc

    def save_model(self, which_model, step):
        assert which_model in ['accuracy', 'weighted_accuracy', 'latest']
        logging.info(f'Improvement achieved: Saving model of client {self.name} on step {step}')
        save_path = Path('./output') / args.outputdirectory / 'models' / f'{self.name}_{which_model}.pth'
        torch.save(self.model.state_dict(), save_path)
        logging.info(f'Saved model of client {self.name} successfully in {save_path}')

    def load_model(self, path: Path, which_model: str) -> None:
        assert which_model in ['accuracy', 'weighted_accuracy', 'latest']
        logging.info(f'Loading global model from {path}')
        load_path = path / f'{self.name}_{which_model}.pth'
        self.update_model(torch.load(load_path, map_location=self._device))
        logging.info(f'Loaded global model successfully from {path / f"{self.name}_{which_model}.pth"}')

    def load_best_model(self, evaluation_metric_name) -> None:
        save_path = Path('./output') / args.outputdirectory / 'models'
        logging.info(f'Loading best model of client {self.name} from {save_path}')
        self.load_model(path=save_path, which_model=evaluation_metric_name)

    def update_wandb_metrics(self, wandb_metrics, train_val_test, loss, accuracy, weighted_accuracy):
        wandb_metrics.update({f'{train_val_test}/{train_val_test}_loss_{self.name}': loss,
                              f'{train_val_test}/{train_val_test}_acc_{self.name}': accuracy,
                              f'{train_val_test}/{train_val_test}_weighted_acc_{self.name}': weighted_accuracy})
