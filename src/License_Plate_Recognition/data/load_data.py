from torch.utils.data import Dataset, DataLoader
from imutils import paths
import albumentations as A
import pandas as pd
import numpy as np
import random
import cv2
import os
from ..misc.separator import *


CHARS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "-",
]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, augment=False):
        self.img_dir = img_dir
        self.img_paths = []
        self.augment = augment
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        print("1dir found, size: ", len(self.img_paths))
        random.shuffle(self.img_paths)
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        height, width, _ = Image.shape

        # if np.random.randint(10) == 2:
        #     Image = cv2.resize(Image,(70,18))
        #     Image = cv2.resize(Image,(94,24))
        Image = cv2.resize(Image, self.img_size)

        # if width/height<2:
        #     Image = bifurcate(Image)
        Image = self.transform(Image)

        basename = os.path.basename(filename)
        imgname, _ = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]
        label = list()
        for c in imgname:
            c = c.upper()
            # one_hot_base = np.zeros(len(CHARS))
            # one_hot_base[CHARS_DICT[c]] = 1
            label.append(CHARS_DICT[c])
        # label = label[:10]
        label_length = len(label)
        # print(label, imgname)
        # if label_length<8 and index!=len(self.img_paths)-1:
        #     Image, label, label_length, filename = self.__getitem__(index+1)
        return Image, label, label_length, filename

    def transform(self, img):
        if self.augment:
            img = self.augment_image(img)
        img = img.astype("float32")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img -= 127.5
        img *= 0.0078125
        # thresh, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        # img = np.reshape(img, img.shape + (1,))
        img = np.transpose(img, (2, 0, 1))
        return img

    def augment_image(self, image):
        transform = A.Compose(
            [
                A.OneOf(
                    [
                        A.IAAAdditiveGaussianNoise(),
                        A.GaussNoise(),
                    ],
                    p=0.3,
                ),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.4),
                        A.MedianBlur(blur_limit=3, p=0.3),
                        A.Blur(blur_limit=3, p=0.3),
                    ],
                    p=0.4,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.IAASharpen(),
                        A.IAAEmboss(),
                        A.RandomBrightnessContrast(),
                    ],
                    p=0.3,
                ),
                A.HueSaturationValue(p=0.3),
            ]
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_image = transform(image=image)["image"]
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        return augmented_image


# class LPRDataLoader(Dataset):                                                         # runs for imagefolder without preprocessing use for testing if no labels (or make preds)
#     def __init__(self, img_dir, imgSize, lpr_max_len, augment=False):
#         self.df = pd.read_csv("ENTIRE_DATASET/labels.csv",encoding= 'unicode_escape')
#         self.img_dir = img_dir
#         self.img_paths = []
#         self.augment = augment
#         for i in range(len(img_dir)):
#             self.img_paths += [el for el in paths.list_images(img_dir[i])]
#         print("1dir found, size: ",len(self.img_paths))
#         random.shuffle(self.img_paths)
#         self.img_size = imgSize
#         self.lpr_max_len = lpr_max_len
#         self.PreprocFun = self.transform

#     def __len__(self):
#         return len(self.img_paths)

#     def __getitem__(self, index):
#         filename = self.img_paths[index]
#         Image = cv2.imread(filename)
#         if Image is None:
#             print(filename)
#         height, width, _ = Image.shape
#         if height != self.img_size[1] or width != self.img_size[0]:
#             Image = cv2.resize(Image, self.img_size)
#         Image = self.transform(Image)

#         basename = os.path.basename(filename)
#         #imgname, _ = os.path.splitext(basename)
#         #imgname = imgname.split("-")[0].split("_")[0]
#         try:
#             target = self.df[self.df.iloc[:,0]==basename].iloc[0,1]
#         except:
#             Image, label, label_length, filename = self.__getitem__(index+1)
#             return Image, label, label_length, filename
#         target = str(target)
#         target = ''.join(e for e in target if e.isalnum())
#         label = list()
#         for c in target:
#             c = c.upper()
#             # one_hot_base = np.zeros(len(CHARS))
#             # one_hot_base[CHARS_DICT[c]] = 1
#             label.append(CHARS_DICT[c])
#         #label = label[:10]
#         label_length = len(label)
#         # if label_length<8 and index!=len(self.img_paths)-1:
#         #     Image, label, label_length, filename = self.__getitem__(index+1)
#         return Image, label, label_length, filename

#     def transform(self, img):
#         img = img.astype('float32')
#         #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         img -= 127.5
#         img *= 0.0078125
#         #thresh, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
#         #img = np.reshape(img, img.shape + (1,))
#         img = np.transpose(img, (2, 0, 1))
#         return img
