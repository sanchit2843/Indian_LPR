import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from albumentations import (
    HueSaturationValue,
    RandomBrightnessContrast,
    Compose,
)

augmentation_pixel_techniques_pool = {
    "RandomBrightnessContrast": RandomBrightnessContrast(
        brightness_limit=(0.005, 0.01), contrast_limit=0.01, p=0.3
    ),
    "HueSaturationValue": HueSaturationValue(
        hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3
    ),
}


class classDataset(Dataset):
    def __init__(self, csvpath, mode, size=(1080, 1920), debug=False):
        """
        Constructor of Dataset
        Format of csv file:
        It contains 2 columns
        1. Path to image
        2. Path to mask

        Parameters:
            csvpath (str): Path to the csv file Ex. dataset/steel
            mode (str): Mode of the dataset {'train', 'valid'}
            height (int): height of the image
            width (int): width of the image
            mean_std (List,List): mean and std of the dataset
            debug (bool): True to show some sample

        """

        self.csv_file = (
            pd.read_csv(os.path.join(csvpath, mode + ".csv")).iloc[:, :].values
        )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.pixel_aug = Compose(
            [
                augmentation_pixel_techniques_pool["RandomBrightnessContrast"],
                augmentation_pixel_techniques_pool["HueSaturationValue"],
            ]
        )

        self.height, self.width = size
        self.mode = mode
        self.debug = debug

    def _set_seed(self, seed):
        """
        Function used to set seed

        Parameters:
            seed (int): seed

        """

        random.seed(seed)
        torch.manual_seed(seed)

    def __len__(self):
        """returns length of CSV file"""
        return len(self.csv_file)

    def __getitem__(self, idx):
        """
        Function used by dataloader to get item contain augmentation, normalization function

        Parameters:
            idx (int): index of the csv file

        """
        image = cv2.imread(self.csv_file[idx, 0], cv2.IMREAD_COLOR)
        label = cv2.imread(self.csv_file[idx, 1], cv2.IMREAD_GRAYSCALE)

        # remove this if n_classes>2
        label[label > 1] = 1
        if (
            image.shape[1] == self.width
            and image.shape[0] == self.height
            and label.shape[1] == self.width
            and label.shape[0] == self.height
        ):
            pass
        else:

            image = cv2.resize(image, (self.width, self.height))
            label = cv2.resize(
                label,
                (self.width, self.height),
                cv2.INTER_NEAREST,
            )

        if self.mode == "train":
            image = self.pixel_aug(image=image)["image"]

        image = self.transform(image)
        label = torch.from_numpy(label)
        sample = {
            "image": image,
            "label": label,
            "img_name": self.csv_file[idx, 0].split("/")[-1],
        }

        return sample
