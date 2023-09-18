import wilds

import pandas as pd
from wilds.datasets.wilds_dataset import WILDSSubset
import numpy as np
import logging

from src.utils import argumentlib
from src.utils.argumentlib import config


class CustomCamelyon(WILDSSubset):
    def __init__(self, transform, csv_file):
        wilds_dataset = wilds.get_dataset(dataset='camelyon17', download=True,
                                          root_dir=config['data_path']['root_data'])
        if isinstance(csv_file, pd.DataFrame):
            self.file = csv_file
        else:
            self.file = pd.read_csv(csv_file)
        indices = self.file['index'].values
        self.labels = self.file.iloc[:, 1:].values.astype(int)
        super().__init__(wilds_dataset, indices, transform)
        logging.info(f'Total # images:{len(indices)} from {csv_file}')

    def __getitem__(self, idx):
        x, y, metadata = super().__getitem__(idx)
        if y.item() == 0:
            y = np.array([1, 0])
        else:
            y = np.array([0, 1])
        return x, y
