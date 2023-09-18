import argparse
import configparser
import logging
import sys
import os

from pathlib import Path

logger = logging.getLogger(__name__)

formatter = logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s')

parser = argparse.ArgumentParser(description='StarAlign Arguments')

os.chdir(sys.path[0])


def parse_args():
    parser.add_argument('--config_file', type=str, default='config_camelyon.ini',
                        help='Configuration filename for the datasets.')

    parser.add_argument('--wandb_projectname', type=str, default=None,
                        help='Project name for wandb logging.'
                             'If None, no wandb logging is used.')

    parser.add_argument('--batch_size', type=int, default=48, help='batch_size per gpu')
    parser.add_argument('--drop_rate', type=int, default=0.2, help='dropout rate')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default='1', help='Which GPU to use. If None, use CPU')
    parser.add_argument('--E', type=int, default=sys.maxsize, help='Number of communication rounds')
    parser.add_argument('--client_number_iterations', type=int, default=100, help='Number of iterations per client')

    # Computational
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for the dataloaders')

    parser.add_argument('--outputdirectory', type=str, default='exper0',
                        help='Name of the output directory. Gets created if it does not exist.')
    parser.add_argument('--validation_interval', type=int, default=1,
                        help='Number of communication rounds before evaluation.')

    parser.add_argument('--aggregation_method', type=str, default='fedbn', choices=['fedavg', 'fedbn'],
                        help='Specify the aggregation method on the server to use. Current choices: fedavg, fedbn')
    parser.add_argument('--equal_weighting', default=False, action=argparse.BooleanOptionalAction,
                        help='When averaging the parameters, weight each client equally')
    parser.add_argument('--adaptation_algorithm_setting', type=str, default=None,
                        help='Specify the adaptation algorithm, target client name and number of samples on target '
                             'client'
                             ' in the format [algorithm]:[target_client_name]:[number_samples]. '
                             'If number of samples is empty, then all of the samples are used.'
                             'E.g. staralign:hospital1:500. Current choices for algorithms: [staralign, default].')

    parser.add_argument('--client_names_training_enabled', nargs='+',
                        help='List of client names that are enabled for training.',
                        default=['hospital0', 'hospital1', 'hospital2', 'hospital3'], type=str)

    parser.add_argument('--optimizer', type=str, default='ADAM', choices=['ADAM', 'SGD'],
                        help='Specify the optimizer to use. Current choices:'
                             'SGD... Stochastic Gradient Descent'
                             'ADAM... ADAM optimizer. Note currently only SGD is supported for StarAlign')

    parser.add_argument('--beta', type=float, default=0.01, help='Beta for StarAlign method')

    parser.add_argument('--models_to_deploy', default=None, type=str,
                        help='Specify the experiment name of the models to deploy. They have to be in the output '
                             'directory')

    parser.add_argument('--oversampling', default=False, action=argparse.BooleanOptionalAction,
                        help='If we use oversampling during training')

    parser.add_argument('--estimate_bn_stats', default=True, action=argparse.BooleanOptionalAction,
                        help='Estimate the running average of mean and variance on clients not present during training')

    args = parser.parse_args()

    # Create the necessary folders
    output_path = Path('./output') / args.outputdirectory
    output_model_path = Path(output_path) / 'models'
    # Create all necessary folders, if they don't exist
    for f in [output_path, output_model_path]:
        os.makedirs(f, exist_ok=True)

    # Set logger
    file_handler = logging.FileHandler(output_path / 'config.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Log arguments
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    return args


def parse_config():
    config = configparser.ConfigParser(allow_no_value=True)
    # Make sure that the keys are not converted to lowercase
    config.optionxform = str
    config.read(Path('./configs') / args.config_file)

    # Log config
    for section in config.sections():
        logger.info("Section: %s", section)
        for options in config.options(section):
            logger.info("x %s:::%s:::%s", options,
                        config.get(section, options), str(type(options)))

    return config


args = parse_args()
config = parse_config()
