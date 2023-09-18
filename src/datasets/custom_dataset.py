from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import torch
from PIL import Image
import logging


class CustomDataset(Dataset):

    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
        :param root_dir: Directory with all the images
        :param csv_file: Path to csv file with annotations
        :param transform: Optional transform to be applied on a sample
        """
        # Also allow pandas dataframe instead of CSV File
        if isinstance(csv_file, pd.DataFrame):
            self.file = csv_file
        else:
            self.file = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.image_paths = self.file['image_path'].values
        self.labels = self.file.iloc[:, 1:].values.astype(int)
        self.transform = transform

        logging.info(f'Total # images:{len(self.image_paths)}, labels:{len(self.labels)}')

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = Path(self.root_dir) / self.image_paths[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_paths)
