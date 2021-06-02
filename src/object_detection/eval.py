import torch
import numpy as np
import cv2
import pandas as pd
import os
from model.fcos import FCOSDetector
from dataloader.custom_dataset import YoloDataset
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from object_detection_metrics_calculation.main import (
    get_coco_metrics_from_path,
)
from object_detection_metrics_calculation.utils import write_txt
import argparse
from utils.utils import sort_by_score, convert_yolotxtline_to_bboxes, preprocess_image


def validate_one_epoch(model, txt_file, output_path, decoder):

    valid_txt = open(txt_file, "r")

    for i, current_line in tqdm(enumerate(valid_txt.readlines())):

        path, boxes, classes = convert_yolotxtline_to_bboxes(current_line)
        frame = cv2.imread(path)
        image = preprocess_image(frame)

        with torch.no_grad():
            out = model(image.cuda())

        # use function to  plot images from txt
        # frame = frame.copy()

        # for bbox in boxes:
        #     frame = cv2.rectangle(
        #         frame,
        #         (bbox[0], bbox[1]),
        #         (bbox[2], bbox[3]),
        #         color=(0, 0, 255),
        #         thickness=2,
        #     )

        # for bbox in out[2][0].cpu().numpy():
        #     frame = cv2.rectangle(
        #         frame,
        #         (int(bbox[0]), int(bbox[1])),
        #         (int(bbox[2]), int(bbox[3])),
        #         color=(255, 0, 0),
        #         thickness=2,
        #     )
        # cv2.imwrite(
        #     os.path.join(output_path, "plotted_images", path.split("/")[-1]),
        #     frame,
        # )

        write_txt(
            (boxes, classes),
            (out[2][0].cpu().numpy(), out[1][0].cpu().numpy(), out[0][0].cpu().numpy()),
            decoder,
            path.split("/")[-1],
            output_path,
        )

    _, all_img_metrics = get_coco_metrics_from_path(os.path.join(output_path))
    return all_img_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--txt_path",
        type=str,
        required=True,
        help="Location to data txt file",
    )

    parser.add_argument(
        "--weight_path",
        type=str,
        required=True,
        help="path to weight pth file",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="output directory to save model",
    )

    args = parser.parse_args()

    decoder = {k: 0 for k in range(10)}

    model = FCOSDetector(mode="inference")

    model.load_state_dict(
        torch.load(
            args.weight_path,
            map_location=torch.device("cpu"),
        )
    )

    model = model.cuda().eval()

    os.makedirs(os.path.join(args.output_path, "groundtruths"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "detections"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "plotted_images"), exist_ok=True)

    all_img_metrics = validate_one_epoch(
        model, args.txt_path, args.output_path, decoder
    )
    print(all_img_metrics)