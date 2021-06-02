import numpy as np
import torch
from torch import nn
import argparse
import os
import cv2
from tqdm import tqdm
import shutil
from models.hrnet import hrnet


from utils.util import (
    overlay_colour,
    plate_locate,
    convert_coordinates_to_bbox,
    get_score_and_class_from_prediction,
    upsample_coordinates,
    postprocess_image,
    preprocess_image,
    convert_yolotxtline_to_bboxes,
)
from object_detection_metrics_calculation.utils import (
    write_txt,
)
from object_detection_metrics_calculation.main import (
    get_coco_metrics_from_path,
)
import matplotlib.pyplot as plt


def validate_one_epoch(model, args, decoder):

    valid_txt = open(args.txt_path, "r")

    for i, current_line in tqdm(enumerate(valid_txt.readlines())):

        path, boxes, classes = convert_yolotxtline_to_bboxes(current_line)
        frame = cv2.imread(path)
        image = preprocess_image(frame).cuda()

        with torch.no_grad():
            prediction = model(image, (image.shape[2], image.shape[3]))

            prediction_softmax = nn.Softmax(dim=1)(prediction["output"])
            prediction = (
                torch.argmax(prediction["output"], dim=1)
                .detach()
                .cpu()
                .squeeze(dim=0)
                .numpy()
                .astype(np.uint8)
            )

            coordinates, centroid = plate_locate(prediction)

            frame = overlay_colour(frame, centroid)

            pred_classes, scores = get_score_and_class_from_prediction(
                prediction, prediction_softmax, coordinates
            )

            pred_boxes = convert_coordinates_to_bbox(coordinates)
            scores_new = []
            pred_boxes_new = []
            for box, score in zip(pred_boxes, scores):
                if score > args.conf_thresh:
                    pred_boxes_new.append(box)
                    scores_new.append(score)
            scores = scores_new
            pred_boxes = pred_boxes_new
            pred_scores = np.asarray(scores)

        for box in pred_boxes:
            frame = cv2.rectangle(
                frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2
            )
        for box in boxes:

            frame = cv2.rectangle(
                frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2
            )

        write_txt(
            (boxes, classes),
            (pred_boxes, pred_classes, pred_scores),
            decoder,
            path.split("/")[-1],
            args.output_path,
        )

        cv2.imwrite(
            os.path.join(args.output_path, "plotted_images", path.split("/")[-1]),
            frame,
        )
    _, all_img_metrics = get_coco_metrics_from_path(os.path.join(args.output_path))
    return all_img_metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--txt_path",
        type=str,
        required=True,
        default="./",
        help="Path to yolo txt file for test data",
    )

    parser.add_argument(
        "--weight_path",
        type=str,
        required=True,
        default="./",
        help="Path to weights for which inference needs to be done",
    )

    parser.add_argument(
        "--conf_thresh",
        type=float,
        required=False,
        default=0.75,
        help="Path to weights for which inference needs to be done",
    )

    parser.add_argument(
        "--n_classes",
        type=int,
        required=False,
        default=2,
        help="Path to weights for which inference needs to be done",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="./",
        help="Path to weights for which inference needs to be done",
    )

    args = parser.parse_args()

    model = hrnet(args.n_classes).eval().cuda()

    # remember to remove this sync batchnorm
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.load_state_dict(torch.load(args.weight_path)["state_dict"])

    os.makedirs(os.path.join(args.output_path, "groundtruths"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "detections"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "plotted_images"), exist_ok=True)

    decoder = {k: 0 for k in range(10)}

    all_image_metrics = validate_one_epoch(model, args, decoder)

    print(all_image_metrics)


if __name__ == "__main__":
    main()