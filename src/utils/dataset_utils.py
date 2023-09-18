import wilds.datasets.wilds_dataset
from torchvision import transforms

from src.datasets.camelyon17 import CustomCamelyon
from src.datasets.custom_dataset import CustomDataset
from src.utils import argumentlib
from src.utils.argumentlib import args, config

import logging

from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from sklearn.model_selection import train_test_split


class ToRangeMinus33(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0.0, 1.0] to a torch.FloatTensor of shape (C x H x W) in the range [-3.0, 3.0]
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return (pic - 0.5) * 6

    def __repr__(self):
        return self.__class__.__name__ + '()'


DATASET_CHARACTERISTICS = {
    'wilds_camelyon': {
        'train_transform': transforms.Compose([transforms.Resize((96, 96)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])]),
        'test_transform': transforms.Compose([transforms.Resize((96, 96)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])
                                              ])
    }
}


def _get_dataset(client_name, train_test_val, transformations, dataset_name):
    """
    Retrieves the clients dataset using the config datapath. Retrieves either train, test or val dataset
    :param client_name: Name of the client
    :param train_test_val: String containing either ['train', 'test', 'val']
    :param transformations: The transforms used for creating the dataset
    :return: The correct dataset
    """
    assert train_test_val.lower() in ['train', 'test', 'val']


    logging.debug(f'Get the federated dataset: {dataset_name}, train, test or val: {train_test_val}')
    if dataset_name == 'wilds_camelyon':
        res_dataset = CustomCamelyon(csv_file=config['clients_csv_paths'][f'{client_name}_{train_test_val}'],
                                     transform=transformations)
    else:
        res_dataset = CustomDataset(root_dir=config['data_path']['root_data'],
                                    csv_file=config['clients_csv_paths'][f'{client_name}_{train_test_val}'],
                                    transform=transformations)

    return res_dataset


def weighted_sampling(target):
    """
    Creates a weighted sampler for the dataloader. This is used for imbalanced datasets
    """
    # Convert the one-hot encoded labels into the corresponding not one-hot encoded class
    target = np.argmax(target, axis=1)
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def get_number_classes(data_set):
    if isinstance(data_set, wilds.datasets.wilds_dataset.WILDSDataset):
        return data_set.n_classes
    elif isinstance(data_set, CustomDataset):
        return data_set.labels.shape[1]
    else:
        raise ValueError(f'The following dataset class is not supported yet {data_set}')


def get_test_dataloaders():
    # The dictionaries for the training and validation datasets and dataloaders
    clients_test_datasets = {}
    clients_test_dataloaders = {}

    dataset_name = config['dataset_attributes']['name']
    transform_test = DATASET_CHARACTERISTICS[dataset_name]['test_transform']

    client_names = config.options('client_names')

    num_classes = []
    for name in client_names:
        # Test
        clients_test_datasets[name] = _get_dataset(client_name=name, train_test_val='test',
                                                   transformations=transform_test, dataset_name=dataset_name)
        clients_test_dataloaders[name] = DataLoader(clients_test_datasets.get(name), batch_size=args.batch_size,
                                                    shuffle=False, num_workers=args.num_workers, pin_memory=False)

        num_classes.append(get_number_classes(clients_test_datasets[name]))

    assert num_classes.count(num_classes[0]) == len(num_classes)
    return clients_test_datasets, clients_test_dataloaders, client_names, num_classes[0]


def prepare_data():
    # The dictionaries for the training and validation datasets and dataloaders
    clients_train_datasets = {}
    clients_val_datasets = {}

    clients_train_dataloaders = {}
    clients_val_dataloaders = {}

    clients_train_datasets_lengths = {}

    logging.info(f'Prepare data from the following config file: {args.config_file}')

    evaluation_metric = config['evaluation_metric']['metric_name']
    assert evaluation_metric in ['accuracy', 'weighted_accuracy']
    dataset_name = config['dataset_attributes']['name']
    # Dynamically get the transforms from the config
    transform_train = DATASET_CHARACTERISTICS[dataset_name]['train_transform']
    transform_test = DATASET_CHARACTERISTICS[dataset_name]['test_transform']

    client_names = config.options('client_names')

    num_classes = []

    for name in client_names:
        # Train
        clients_train_datasets[name] = _get_dataset(client_name=name, train_test_val='train',
                                                    transformations=transform_train, dataset_name=dataset_name)

        num_classes.append(get_number_classes(clients_train_datasets[name]))
        # Also add length of dataset (this is used for 'weighting' during FedAvg)
        clients_train_datasets_lengths[name] = len(clients_train_datasets[name])
        if args.oversampling:
            # Sampler is mutually exclusive with shuffle=True. Therefore, no shuffling
            clients_train_dataloaders[name] = DataLoader(clients_train_datasets.get(name), batch_size=args.batch_size,
                                                         num_workers=args.num_workers, pin_memory=False,
                                                         sampler=weighted_sampling(
                                                             clients_train_datasets.get(name).labels))
        else:
            clients_train_dataloaders[name] = DataLoader(clients_train_datasets.get(name), batch_size=args.batch_size,
                                                         shuffle=True, num_workers=args.num_workers, pin_memory=False)
        # Validation
        clients_val_datasets[name] = _get_dataset(client_name=name, train_test_val='val',
                                                  transformations=transform_test,
                                                  dataset_name=dataset_name)
        clients_val_dataloaders[name] = DataLoader(clients_val_datasets.get(name), batch_size=args.batch_size,
                                                   shuffle=False, num_workers=args.num_workers, pin_memory=False)

    if args.adaptation_algorithm_setting is not None:
        _, target_client_name, limlab_num_samples = args.adaptation_algorithm_setting.split(':')
        # Create limited labels if in arguments
        if limlab_num_samples.isdigit():
            logging.info(f'Create limited labels for client {target_client_name} with {limlab_num_samples} samples')
            clients_train_dataloaders[target_client_name], clients_train_datasets[target_client_name] = \
                create_limited_labels_dataloader(dataset=clients_train_datasets[target_client_name],
                                                 number_examples=int(limlab_num_samples))
            # Update client dataset lengths
            clients_train_datasets_lengths[target_client_name] = int(limlab_num_samples)
        else:
            logging.info(f'No limited labels are created and limlab_num_samples is: {limlab_num_samples}')
            assert limlab_num_samples == ''

    # Check if there are duplicates in num_classes list
    assert num_classes.count(num_classes[0]) == len(num_classes)
    return (num_classes[0], clients_train_datasets_lengths, clients_train_datasets, clients_train_dataloaders,
            clients_val_datasets, clients_val_dataloaders, evaluation_metric)


def create_limited_labels_dataloader(dataset, number_examples, stratify=True):
    # NOTE: Dataset are not shuffled, therefore I can create these indices
    if stratify:
        targets = dataset.labels
        # Convert it to Pytorch Tensor
        targets = torch.from_numpy(targets).long()
        # Convert to not one_hot
        targets = torch.argmax(targets, dim=1)
        subset_complement, subset = train_test_split(np.arange(len(targets)), test_size=number_examples,
                                                     random_state=args.seed, shuffle=True, stratify=targets)

    else:
        subset = range(0, number_examples)
    limited_labels_trainset = torch.utils.data.Subset(dataset, subset)

    if args.oversampling:
        # Extract the labels from the dataset with the corresponding indices
        limited_labels_labels = dataset.labels[limited_labels_trainset.indices]
        limited_labels_trainloader = DataLoader(limited_labels_trainset, batch_size=args.batch_size,
                                                num_workers=args.num_workers, pin_memory=False,
                                                sampler=weighted_sampling(limited_labels_labels))
    else:
        limited_labels_trainloader = DataLoader(limited_labels_trainset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.num_workers, pin_memory=False)

    return limited_labels_trainloader, limited_labels_trainset
