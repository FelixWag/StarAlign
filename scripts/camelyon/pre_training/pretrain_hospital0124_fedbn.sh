#!/bin/bash

CONDA_ENV="staralign"
EXPERIMENT_NAME="PRETRAIN_H0124_FedBN_sanity"

# Activate conda env if in base env, or don't if already set.
source "$(dirname $(dirname $(which conda)))/etc/profile.d/conda.sh"
if [[ "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]]; then
  echo "activating ${CONDA_ENV} env"
  set +u; conda activate "${CONDA_ENV}"; set -u
fi

# Run the experiment
python3 ../../../main.py --outputdirectory $EXPERIMENT_NAME --gpu 0 --config "config_camelyon.ini" --wandb_projectname "fl-fish_camelyon" --E 350 --lr 1e-3 --optimizer "SGD" --batch_size 32 --equal_weighting --client_names_training_enabled 'hospital0' 'hospital1' 'hospital2' 'hospital4' --aggregation_method 'fedbn'


conda deactivate