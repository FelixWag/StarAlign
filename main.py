# First change the working directory!
import os
import sys

from src.utils import argumentlib
from src.utils.argumentlib import args, config

from src.federated.client import Client
from src.federated.server import Server
from src.utils.dataset_utils import prepare_data, get_test_dataloaders
import logging
import random
import numpy as np
import copy
import torch
import wandb
import torch.backends.cudnn as cudnn
from torchvision.models import densenet121
from torch import nn

from pathlib import Path
from tqdm import tqdm

os.chdir(sys.path[0])

if __name__ == '__main__':

    wandb_metrics = {}

    if args.wandb_projectname is not None:
        wandb.init(project=args.wandb_projectname, name=args.outputdirectory, config=args)

    criterion = nn.CrossEntropyLoss()

    output_path = Path('./output') / args.outputdirectory

    logging.basicConfig(filename=os.path.join(output_path, 'log.txt'), level=logging.INFO,
                        format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
    # Also add to StdOut
    logFormatter = logging.Formatter('[%(asctime)s] %(message)s')
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logging.getLogger().addHandler(consoleHandler)

    logging.info(f"CUDA is available: {torch.cuda.is_available()}")
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    criterion = criterion.to(device)

    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # Get all datasets (also creates the limited labels dataloaders)
    (num_classes, clients_train_datasets_lengths, clients_train_datasets, clients_train_dataloaders,
     clients_val_datasets, clients_val_dataloaders, evaluation_metric) = prepare_data()
    logging.info(f'Following evaluation metric is used {evaluation_metric}')
    _, clients_test_dataloaders, _, _ = get_test_dataloaders()
    client_names = config.options('client_names')

    net_glob = densenet121(drop_rate=args.drop_rate, num_classes=num_classes)

    # Sanity check
    assert set(args.client_names_training_enabled).issubset(set(client_names))

    if args.adaptation_algorithm_setting is not None:
        training_algorithm, target_client_name, _ = args.adaptation_algorithm_setting.split(':')
        assert training_algorithm in ['staralign', 'default']
        assert target_client_name in client_names
    else:
        training_algorithm = 'default'
        target_client_name = None

    assert args.client_names_training_enabled is not None
    server = Server(global_model=net_glob, global_model_template=copy.deepcopy(net_glob),
                    client_names_training_enabled=args.client_names_training_enabled,
                    aggregation_method=args.aggregation_method, device=device,
                    training_algorithm=training_algorithm, target_client_name=target_client_name)

    for client_name in client_names:
        if args.optimizer == 'ADAM':
            optimizer_class = torch.optim.Adam
            optimizer_params = {'lr': args.lr, 'betas': (0.9, 0.999), 'weight_decay': 5e-4}
        elif args.optimizer == 'SGD':
            optimizer_class = torch.optim.SGD
            optimizer_params = {'lr': args.lr}
        else:
            raise ValueError(f'Invalid optimizer: {args.optimizer}')

        client = Client(model=copy.deepcopy(net_glob), optimizer_class=optimizer_class,
                        optimizer_params=optimizer_params,
                        dataset_length=clients_train_datasets_lengths[client_name], name=client_name, device=device,
                        train_dataloader=clients_train_dataloaders[client_name],
                        val_dataloader=clients_val_dataloaders[client_name],
                        test_dataloader=clients_test_dataloaders[client_name],
                        server=server, criterion=criterion)

        server.add_client(client=client)

    if args.models_to_deploy is not None:
        checkpoint_path = Path('./output') / args.models_to_deploy / 'models'
        server.load_global_and_client_models(model_directory_path=checkpoint_path,
                                             which_model='latest')

    # ------------------------------- Evaluate before training the model -----------------------------------------------
    server.evaluate(train_test_val='val', num_classes=num_classes, step=0, save_best_model=False,
                    wandb_metrics=wandb_metrics)
    if args.wandb_projectname is not None:
        wandb.log(wandb_metrics, step=0)
    # ---------------------------------------- End Evaluation ----------------------------------------------------------

    for com_round in tqdm(range(args.E), position=0):
        # Server starts training the clients
        server.train()

        # ------------------------------------ Evaluate ----------------------------------------------------------------
        if (com_round + 1) % args.validation_interval == 0:
            server.evaluate(train_test_val='val', num_classes=num_classes, step=(com_round + 1),
                            wandb_metrics=wandb_metrics, save_latest=True)
            if args.wandb_projectname is not None:
                wandb.log(wandb_metrics, step=(com_round + 1) * args.client_number_iterations)

    # Load best model
    server.load_best_global_and_client_models(evaluation_metric_name=evaluation_metric)
    # ---------- Evaluate on test set ---------------
    server.evaluate(train_test_val='test', num_classes=num_classes, step=None, wandb_metrics=wandb_metrics,
                    save_best_model=False)

    if args.wandb_projectname is not None:
        wandb.finish()
