import statistics

import torch
import torch.nn as nn

from src.federated.client import Client
from src.federated.federated_algorithms import fed_avg, fed_bn
from src.utils import argumentlib
from src.utils.argumentlib import args, config
import logging

from pathlib import Path


class Server:
    def __init__(self, global_model: torch.nn.Module, global_model_template: torch.nn.Module,
                 client_names_training_enabled: list[str], aggregation_method: str, device,
                 training_algorithm: str = 'default', target_client_name: str = None):
        self._global_model = global_model
        self.all_clients = {}
        self.training_clients = {}
        self._client_names_training_enabled = client_names_training_enabled
        self._training_algorithm = training_algorithm
        self._target_client_name = target_client_name
        # This is used to get the model architecture of the global model but does not update the weights
        self._global_model_template = global_model_template
        self._aggregation_method = aggregation_method

        self._best_selection_metric = 0.0

        self._device = device
        # Move both models to device
        self._global_model.to(device)
        self._global_model_template.to(device)

    @property
    def global_model(self) -> nn.Module:
        return self._global_model

    @property
    def global_model_template(self) -> nn.Module:
        return self._global_model_template

    @property
    def global_model_weights(self) -> dict:
        return self._global_model.state_dict()

    @property
    def target_client(self) -> nn.Module:
        assert self._target_client_name is not None
        return self.all_clients[self._target_client_name]

    @property
    def target_client_name(self) -> str:
        return self._target_client_name

    def update_model(self, new_model) -> None:
        if isinstance(new_model, nn.Module):
            self._global_model.load_state_dict(new_model.state_dict())
        elif isinstance(new_model, dict):
            self._global_model.load_state_dict(new_model)
        else:
            raise ValueError('Invalid input: new_model must be an instance of nn.Module or a state dictionary.')

    def add_client(self, client: Client) -> None:
        assert client.name not in self.all_clients
        self.all_clients[client.name] = client
        if client.name in self._client_names_training_enabled:
            self.training_clients[client.name] = client

    def train(self) -> None:
        losses = []
        # First, local training on clients
        for _, client in self.training_clients.items():
            loss = client.train(self._training_algorithm)
            losses.append(loss)
        logging.info(f'Average loss on clients: {statistics.fmean(losses):.2f}')
        # Aggregate the client model and update global model
        with torch.no_grad():
            if self._aggregation_method == 'fedavg':
                w_avg = fed_avg(server=self, clients=self.training_clients, equal_weighting=args.equal_weighting)
                # Update/Send the client models
                for _, client in self.all_clients.items():
                    client.update_model(w_avg)
            elif self._aggregation_method == 'fedbn':
                w_avg, w_avg_non_norm_params = fed_bn(server=self, clients=self.training_clients,
                                                      equal_weighting=args.equal_weighting)
                for _, client in self.all_clients.items():
                    # If it is in training, then update with non_norm_params, to keep them local
                    if client.name in self._client_names_training_enabled:
                        # We need to set strict to false since we are not loading the normalization layers
                        client.update_model(w_avg_non_norm_params, strict=False)
                    else:
                        # Update rest of the models with the server model
                        client.update_model(w_avg)
                        if args.estimate_bn_stats:
                            client.estimate_bn_stats('val')

            else:
                raise ValueError(f"Unsupported aggregation method: {self._aggregation_method}")

    def evaluate(self, train_test_val, num_classes, step, wandb_metrics, save_best_model=True, save_latest=False) -> None:
        loss_all = []
        correct_all = 0
        total_all = 0

        client_accuracies = []
        client_weighted_accuracies = []

        for _, client in self.all_clients.items():
            loss, correct, total, weighted_accuracy = client.evaluate(train_test_val=train_test_val,
                                                                      num_classes=num_classes, step=step,
                                                                      save_best_model=save_best_model,
                                                                      wandb_metrics=wandb_metrics,
                                                                      save_latest=save_latest)
            loss_all.append(loss)
            correct_all += correct
            total_all += total
            client_accuracies.append(correct / total)
            client_weighted_accuracies.append(weighted_accuracy)

        # Compute mean loss and accuracy
        mean_loss = statistics.fmean(loss_all)
        mean_total_acc = correct_all / total_all
        mean_acc = statistics.fmean(client_accuracies)
        mean_weighted_acc = statistics.fmean(client_weighted_accuracies)

        # Log to console and wandb
        logging.info(
            f'Evaluate server on step {step}. Mean loss is: {mean_loss:.2f}, Mean accuracy is: {mean_acc:.2%}, '
            f'Mean total accuracy is: {mean_total_acc:.2%}, Mean weighted accuracy is: {mean_weighted_acc:.2%}')
        if args.wandb_projectname is not None:
            self.update_wandb_metrics(wandb_metrics=wandb_metrics, train_val_test=train_test_val,
                                      loss=mean_loss, accuracy=mean_total_acc, mean_accuracy=mean_acc,
                                      weighted_accuracy=mean_weighted_acc)

        if save_best_model:
            if config['evaluation_metric']['metric_name'] == 'accuracy':
                metric = mean_total_acc
            elif config['evaluation_metric']['metric_name'] == 'weighted_accuracy':
                metric = mean_weighted_acc
            else:
                raise ValueError(f'Invalid evaluation metric: {config["evaluation_metric"]["metric_name"]}')
            if metric > self._best_selection_metric:
                self._best_selection_metric = metric
                self.save_model(which_model=config['evaluation_metric']['metric_name'], step=step)
        if save_latest:
            self.save_model(which_model='latest', step=step)

    def save_model(self, which_model, step) -> None:
        assert which_model in ['accuracy', 'weighted_accuracy', 'latest']
        logging.info(f'Improvement achieved: Saving global model on step {step}')
        save_path = Path('./output') / args.outputdirectory / 'models' / f'global_{which_model}.pth'
        torch.save(self.global_model.state_dict(), save_path)
        logging.info(f'Saved global model successfully in {save_path}')

    def load_global_model(self, model_directory_path: Path, which_model: str) -> None:
        assert which_model in ['accuracy', 'weighted_accuracy', 'latest']
        logging.info(f'Loading global model from {model_directory_path}')
        load_path = model_directory_path / f'global_{which_model}.pth'
        self.update_model(torch.load(load_path, map_location=self._device))
        logging.info(
            f'Loaded global model successfully from {model_directory_path / f"global_{which_model}.pth"}')

    def load_best_global_model(self, evaluation_metric_name) -> None:
        save_path = Path('./output') / args.outputdirectory / 'models'
        logging.info(f'Loading best global model')
        self.load_global_model(model_directory_path=save_path, which_model=evaluation_metric_name)

    def load_best_global_and_client_models(self, evaluation_metric_name) -> None:
        self.load_best_global_model(evaluation_metric_name=evaluation_metric_name)
        for _, client in self.all_clients.items():
            client.load_best_model(evaluation_metric_name=evaluation_metric_name)

    def load_global_and_client_models(self, model_directory_path: Path, which_model: str) -> None:
        self.load_global_model(model_directory_path=model_directory_path, which_model=which_model)
        for _, client in self.all_clients.items():
            client.load_model(path=model_directory_path, which_model=which_model)

    def update_wandb_metrics(self, wandb_metrics, train_val_test, loss, accuracy, mean_accuracy, weighted_accuracy):
        wandb_metrics.update({f'{train_val_test}/{train_val_test}_acc_all': accuracy,
                              f'{train_val_test}/{train_val_test}_acc_all_mean': mean_accuracy,
                              f'{train_val_test}/{train_val_test}_loss_all': loss,
                              f'{train_val_test}/{train_val_test}_weighted_acc_all': weighted_accuracy
                              })
