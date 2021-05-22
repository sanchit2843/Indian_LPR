import os
import pandas as pd
import numpy as np

images = []
for i in os.listdir(
    "/media/sanchit/datasets/Our-collected-dataset/plate_data_download/ocr_car_plates_only/images"
):
    images.append(np.asarray([i, i.split("_")[0]]))
images = np.asarray(images)
print(images.shape)
pd.DataFrame(images).to_csv(
    "/media/sanchit/datasets/Our-collected-dataset/plate_data_download/ocr_car_plates_only/ocr.csv",
    index=False,
)
