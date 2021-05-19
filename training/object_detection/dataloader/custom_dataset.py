import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from albumentations import (
    HueSaturationValue,
    RandomBrightnessContrast,
    Compose,
)


augmentation_pixel_techniques_pool = {
    "RandomBrightnessContrast": RandomBrightnessContrast(
        brightness_limit=(0.005, 0.01), contrast_limit=0.01, p=0.1
    ),
    "HueSaturationValue": HueSaturationValue(
        hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.1
    ),
}


class YoloDataset(Dataset):
    def __init__(self, text_file, train=True):

        text_file = open(text_file, mode="r+")
        self.list_sample = text_file.readlines()[::-1]
        self.train = train
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

    def __len__(self):
        return len(self.list_sample)

    def __getitem__(self, idx, debug=False):
        current_line = self.list_sample[idx]
        image = cv2.imread(current_line.split(" ")[0])

        if self.train:
            image = self.pixel_aug(image=image)["image"]
        boxes, classes = self.process_txt(current_line)

        if debug:
            self.debug(image, boxes)
        return (
            self.transform(image),
            torch.from_numpy(np.asarray(boxes)),
            torch.from_numpy(np.asarray(classes)),
            current_line.split(" ")[0],
        )

    def process_txt(self, current_line):
        boxes = []
        classes = []

        if len(current_line.split(" ")[1:][0]) > 6:
            for bbox in current_line.split(" ")[1:]:
                a = []
                for idx, x in enumerate(bbox.split(",")):
                    if idx == 4:
                        classes.append(int(x))
                        break
                    if idx % 2 == 0:
                        a.append(max(0, int(int(x))))
                    else:
                        a.append(max(0, int(int(x))))
                boxes.append(a)
        return boxes, classes

    def debug(self, image, boxes):
        image1 = image.copy()
        for bbox in boxes:
            image1 = cv2.rectangle(
                image1,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color=(255, 0, 0),
                thickness=2,
            )
        plt.imshow(image1[:, :, ::-1])
        plt.show()

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list, paths = zip(*data)
        batch_size = len(boxes_list)

        pad_boxes_list = []
        pad_classes_list = []
        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num:
                max_num = n

        for i in range(batch_size):
            if len(boxes_list[i]) != 0:
                pad_boxes_list.append(
                    torch.nn.functional.pad(
                        boxes_list[i],
                        (0, 0, 0, max_num - boxes_list[i].shape[0]),
                        value=-1,
                    )
                )
                pad_classes_list.append(
                    torch.nn.functional.pad(
                        classes_list[i],
                        (0, max_num - classes_list[i].shape[0]),
                        value=-1,
                    )
                )
            else:
                pad_boxes_list.append(torch.ones(max_num, 5) * -1)
                pad_classes_list.append(torch.ones(max_num) * -1)
        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(imgs_list)
        return batch_imgs, batch_boxes, batch_classes, paths


if __name__ == "__main__":

    train_dataset = YoloDataset(
        "/media/sanchit/datasets/Our-collected-dataset/plate_data_download/Data_public/train_yolo.txt",
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
    )
    for i in range(len(train_dataset)):
        _ = train_dataset.__getitem__(i, True)
