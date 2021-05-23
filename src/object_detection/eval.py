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
from utils.utils import sort_by_score


def validate_one_epoch(model, eval_loader, output_path, decoder):
    gt_boxes = []
    gt_classes = []
    pred_boxes = []
    pred_classes = []
    pred_scores = []

    for i, [img, boxes, classes, paths] in tqdm(enumerate(eval_loader)):

        with torch.no_grad():
            out = model(img.cuda())
            pred_boxes.append(out[2][0].cpu().numpy())
            pred_classes.append(out[1][0].cpu().numpy())
            pred_scores.append(out[0][0].cpu().numpy())

        gt_boxes.append(boxes[0].numpy())
        gt_classes.append(classes[0].numpy())
        pred_boxes, pred_classes, pred_scores = sort_by_score(
            pred_boxes, pred_classes, pred_scores
        )

        frame = frame.copy()
        for bbox in boxes[0].numpy():
            frame = cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color=(0, 0, 255),
                thickness=4,
            )
        for bbox in out[2][0].cpu().numpy():
            frame = cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color=(255, 255, 0),
                thickness=3,
            )
        cv2.imwrite(
            os.path.join(output_path, "plotted_images", paths[0].split("/")[-1]),
            frame,
        )

        write_txt(
            (boxes[0].numpy(), classes[0].numpy()),
            (out[2][0].cpu().numpy(), out[1][0].cpu().numpy(), out[0][0].cpu().numpy()),
            decoder,
            paths[0].split("/")[-1],
            output_path,
        )

        pred_boxes = []
        pred_scores = []
        pred_classes = []
        gt_boxes = []
        gt_classes = []

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

    eval_dataset = YoloDataset(
        args.txt_path,
        train=False,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False, collate_fn=eval_dataset.collate_fn
    )

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

    all_img_metrics = validate_one_epoch(model, eval_loader, args.output_path, decoder)
