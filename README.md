# ⭐ StarAlign: Post-Deployment Adaptation with Access to Source Data via Federated Learning and Source-Target Remote Gradient Alignment

---

This is the official PyTorch implementation of our paper, Wagner et al.,  [Post-Deployment Adaptation with Access to Source Data via Federated Learning and Source-Target Remote Gradient Alignment](https://arxiv.org/abs/2308.16735), Machine Learning for Medical Imaging (MLMI) - MICCAI 2023. If you find this code useful for your research, please cite our paper, the BibTex version can be found [at the end](#citation) of the README.

## Introduction

---

Distribution shift between source and target data negatively impacts performance of deployed Deep Neural Networks. Post Deployment Adaptation/Test-Time adaptation methods tailor a pre-trained model to a specific target distribution and assume no access to source data because of privacy concerns or large size. They adapt with minimal labeled or unlabeled target data, which provides only limited learning signal. Federated Post-Deployment Domain Adaptation (**FedPDA**) enables a deployed model to access source data through Federated Learning and adapt it for a target distribution by extracting relevant information for the target data from the source clients. 

We propose (**S**ource-**Tar**get **R**emote Gradient **Align**ment) ⭐**StarAlign**⭐, a novel Federated Post-Deployment Domain Adaptation method that aligns the gradients of the source and target client pairs to extract relevant information for the target data. This repository contains the implementation of our StarAlign algorithm. Furthermore, this repository provides an implementation of FedAvg [2] and FedBN [3] and can be used as FedPDA framework.

![Staralign Framework](assets/staralignfigure.png?raw=true "Optional Title")

## Installation

---

### Dependencies
We recommend using **conda** for installing the dependencies. The following command will create a new environment with all the required dependencies:
```
conda env create -f environment.yml
conda activate staralign
```
Alternatively, you can install the dependencies manually. The code depends on the following packages (tested on the following versions):
* ``Python`` version 3.11
* ``numpy`` version 1.25.2
* ``pillow`` version 9.4.0
* ``pytorch`` version 2.0.1
* ``torchvision`` version 0.15.2
* ``tqdm`` version 4.66.1
* ``typing_extensions`` version 4.7.1
* ``pandas`` version 2.1.0
* ``pathtools`` version 0.1.2
* ``scikit-learn`` version 1.3.0
* ``wandb`` version 0.15.10
* ``wilds`` version 2.0.0

## Usage

---
To start running experiments and train models, you can use the `main.py` script by executing the following command:

```bash
python main.py --outputdirectory EXPERIMENT_NAME --gpu 0 --config "config_camelyon.ini" --E 350 --lr 1e-3 --optimizer "SGD" --batch_size 32 --equal_weighting --client_names_training_enabled 'hospital0' 'hospital1' 'hospital2' 'hospital3' --aggregation_method 'fedbn'
```

An overview of most relevant arguments is given below:

|            Parameter            | Description                                                                                                                                                                                                                                                                                                                                                                |
|:-------------------------------:|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|        `outputdirectory`        | Name of the experiment. Files will be saved in `output/outputidrectory`                                                                                                                                                                                                                                                                                                    |
|            `config`             | Specifies the name of the dataset configuration file. It is an `INI` file and has to be stored in `/configs`. You can find sample configuration files in this folder. More information in the dataset section.                                                                                                                                                             |
|               `E`               | Number of communication rounds                                                                                                                                                                                                                                                                                                                                             |
|              `lr`               | Learning rate                                                                                                                                                                                                                                                                                                                                                              |
|             `beta`              | &beta; parameter for staralign method                                                                                                                                                                                                                                                                                                                                      |
|         `oversampling`          | If this flag is set, oversampling is used (useful for imbalanced datasets).                                                                                                                                                                                                                                                                                                |
|           `optimizer`           | Which optimizer to use. Currently supported: `[SGD, ADAM]`                                                                                                                                                                                                                                                                                                                 |
|        `equal_weighting`        | If this flag is set, the server weights each client model equally during the averaging step. If the flag is not set, the server weights each model according to its dataset size during the averaging step.                                                                                                                                                                |
| `client_names_training_enabled` | List a name of clients that participate in the training loop. They must match the names in the corresponding config file. E.g.: `--client_names_training_enabled 'hospital0' 'hospital1' 'hospital2'`                                                                                                                                                                      |
|      `aggregation_method`       | Specify the aggregation method on the server. Current choices: `['fedavg','fedbn']`                                                                                                                                                                                                                                                                                        |
| `adaptation_algorithm_setting`  | Specify the adaptation algorithm, target client name and number of samples on target client in the format `[algorithm]:[target_client_name]:[number_samples]`. If number of samples is empty, then all of the samples are used.E.g. staralign:hospital1:500. Current choices for algorithms: [staralign, default]. `default` means standard local training on each client. |
|       `models_to_deploy`        | Specify the experiment name of the models to deploy. They have to be in the `./output/` directory.                                                                                                                                                                                                                                                                         |


To get a description of all available arguments, you can use the following command:

```bash
python main.py --help
```

In the `./scripts` folder, some example scripts are provided. You can use them as a starting point for your own experiments. 
For example, to run one of the example scripts, you can use the following command in your terminal:

```bash
/scripts/camelyon/pre_training/pretrain_hopsital0123_fedbn.sh
```

### Pre-training models

To pretrain models, include the client names you want to be trained in the `client_names_training_enabled` argument. For example, if you want to use the Camelyon17 dataset from WILDS [1] and want to pretrain the models on `hospital0`, `hospital1`, `hospital2`, and `hospital4`, you can use the following command:

```bash
python main.py --outputdirectory 'PRETRAIN_H0124' --gpu 0 --config "config_camelyon.ini" --E 350 --lr 1e-3 --optimizer "SGD" --batch_size 32 --equal_weighting --client_names_training_enabled 'hospital0' 'hospital1' 'hospital2' 'hospital4' --aggregation_method 'fedbn'
```

This will store the pretrained models in the `./output/PRETRAIN_H0124` directory.

### Adapting models

You can then use the `models_to_deploy` argument to specify the experiment name of the models to deploy. Furthermore, with the `adaptation_algorithm_setting` you can specify a specific adaptation algorithm, target client name and number of samples on target client. In the following format `[algorithm]:[target_client_name]:[number_samples]` (e.g. `staralign:hospital1:500`). If number of samples is empty, then all of the samples are used. Current choices for algorithms: `[staralign, default]`. `default` means standard local training on each client.

For example, if you want to **deploy** and **adapt** the models trained in the `PRETRAIN_H0124` experiment to the target client `hospital3`, you can use the following command:

```bash
python main.py --outputdirectory 'ADAPT_H3_staralign' --gpu 0 --config "config_camelyon.ini" --E 50 --lr 0.1 --beta 0.2 --optimizer "SGD" --batch_size 32 --equal_weighting --client_names_training_enabled 'hospital0' 'hospital1' 'hospital2' 'hospital3' 'hospital4' --aggregation_method 'fedbn' --adaptation_algorithm_setting "staralign:hospital3:1558" --models_to_deploy "PRETRAIN_H0124"
``` 

## Datasets

---

In general, to run experiments on a new dataset you need one configuration file and csv files for each client. 

### CSV file structure
For each client in the dataset, it is required to have three CSV files, each representing training, testing and validation data. These CSV files should have the following structure:

- **Column 1**: `image_name` (String). This column should contain the names of the images.
- **Remaining Columns**: One-Hot Encoded Classes (0 or 1): The rest of the columns should represent the classes as one-hot encoded values, where '0' indicates absence and '1' indicates presence of the respective class.

Here is an example of what the CSV files should look like:

```csv
image_name,class_A,class_B,class_C
image_0.png,0,1,0
image_1.png,1,0,0
image_2.png,0,0,1
```

**Note:** We provide the csv file for Camelyon17 dataset from WILDS [1] in the `./data` folder. Please be aware that it has a structure that differs slightly from the one described above because we are using the data from the WILDS GitHub repository [4].

Each dataset must be specified in a configuration file (containing the file paths). The configuration file is an `INI` file and has the following structure:

### Config file structure
The config file should be places in the `./configs` folder and must be specified in the `--config` argument when executing the `main.py` script. It must have the following sections:

|      Section name      | Keys and values                                                                                                                                                                                                                                                                                                                                                                                                         | Description                                                                                                                                                                                  |
|:----------------------:|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|     `[data_path]`      | The key must be `root_data` and the value must be the path where the images are stored: E.g. `root_data = /path/to/images`                                                                                                                                                                                                                                                                                              | This section defines the location where the actual images are stored. It is a crucial setting for the application as it specifies the directory where the program will look for image files. |
| `[clients_csv_paths]`  | Each client should have corresponding keys that match their names in the `[client_names]` section, followed by '_train', '_test', or '_val' to specify the dataset type. The keys must be the path to the corresponding file. **Example**: <ul> <li> `client1_train = /path/to/client1_train.csv` </li> <li> `client1_test = /path/to/client1_test.csv` </li> <li> `client1_val = /path/to/client1_val.csv` </li> </ul> | This section is used to define the paths to CSV files for training, testing, and validation datasets for different clients.                                                                  |
|    `[client_names]`    | **Note:** This section only requires keys! No values are needed. The keys represent the name of each client                                                                                                                                                                                                                                                                                                             | The names of each client                                                                                                                                                                     |
| `[dataset_attributes]` | The key should be `name` and the value should be the name of the dataset. E.g.: `name = wilds_camelyon`                                                                                                                                                                                                                                                                                                                 | This should be the name of the dataset                                                                                                                                                       |
| `[evaluation_metric]`  | The key should be `metric_name` and the value should be name of the metric. Currently supported choices are: `[weighted_accuracy, accuracy]`                                                                                                                                                                                                                                                                            | This is used to save the best model on the validation set using the specified metric. Both metrics are computed and displayed during training                                                |

A sample config file for the camleyon dataset can be found in `./configs/config_camelyon.ini`.


### Datasets in this repository
This repository includes and offers easy access to the Camelyon17 dataset from WILDS [1]. The dataset is split into 5 hospitals. The dataset configuration file `config_camelyon.ini` is provided in the `./configs` folder. You can use this file to specify the dataset configuration.


## Citation

---

```
@misc{wagner2023postdeployment,
      title={Post-Deployment Adaptation with Access to Source Data via Federated Learning and Source-Target Remote Gradient Alignment}, 
      author={Felix Wagner and Zeju Li and Pramit Saha and Konstantinos Kamnitsas},
      year={2023},
      eprint={2308.16735},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## References

---
[1] P. W. Koh et al., “WILDS: A benchmark of in-the-wild distribution shifts” 2020, arXiv:2012.07421

[2] H. B. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" 2016, arxiv:1602.05629

[3] X. Li et al., "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization" 2021, ICLR

[4] https://github.com/p-lambda/wilds

## Contact

**Felix Wagner**

I hope you find this code useful and valuable! Your feedback, comments, and suggestions are highly appreciated.

If you have any questions, encounter issues, or want to share your thoughts, please don't hesitate to reach out:

📧 Email: felix.wagner (AT) eng.ox.ac.uk

## License
This project is licensed under the [MIT license](LICENSE)

Copyright (c) 2023 Felix Wagner