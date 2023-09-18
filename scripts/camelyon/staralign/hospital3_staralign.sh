#!/bin/bash

CONDA_ENV="staralign"
EXPERIMENT_NAME="H3_adapt_staralign_third"

# Activate conda env if in base env, or don't if already set.
source "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]]; then
  echo "activating ${CONDA_ENV} env"
  set +u; conda activate "${CONDA_ENV}"; set -u
fi

python3 ../../../main.py --outputdirectory $EXPERIMENT_NAME --gpu 1 --config "config_camelyon.ini" --wandb_projectname "fl-fish_camelyon" --E 50 --lr 0.1 --beta 0.2 --optimizer "SGD" --batch_size 32 --equal_weighting --client_names_training_enabled 'hospital0' 'hospital1' 'hospital2' 'hospital3' 'hospital4' --aggregation_method 'fedbn' --adaptation_algorithm_setting "staralign:hospital3:1558" --models_to_deploy "PRETRAIN_H0124_FedBN_sanity"

conda deactivate