import os
import cv2
from albumentations import (
    HueSaturationValue,
    RandomBrightnessContrast,
    Compose,
)


augmentation_pixel_techniques_pool = {
    "RandomBrightnessContrast": RandomBrightnessContrast(
        brightness_limit=(0.05, 0.1), contrast_limit=0.1, p=1
    ),
    "HueSaturationValue": HueSaturationValue(
        hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3
    ),
}

pixel_aug = Compose(
    [
        # augmentation_pixel_techniques_pool["RandomBrightnessContrast"],
        augmentation_pixel_techniques_pool["HueSaturationValue"],
    ]
)
image = cv2.imread(
    "/media/sanchit/datasets/Our-collected-dataset/plate_data_download/Data_public/Images_png/NO20200731-071935-000025-converted_3.png"
)
import matplotlib.pyplot as plt

plt.imshow(image[:, :, ::-1])
plt.show()
image = pixel_aug(image=image)["image"]
plt.imshow(image[:, :, ::-1])
plt.show()
