import numpy as np
import torch
from torch import nn
import argparse
import os
import cv2
from tqdm import tqdm
import shutil
from src.semantic_segmentation.models.hrnet import hrnet
from src.semantic_segmentation.utils.yolo_dataset import YoloDataset
from src.semantic_segmentation.utils.util import (
    overlay_colour,
    plate_locate,
    convert_coordinates_to_bbox,
    get_score_from_prediction,
    upsample_boxes,
)
from src.semantic_segmentation.object_detection_metrics_calculation.utils import (
    write_txt,
)


def validate_one_epoch(model, eval_loader, args, decoder):
    gt_boxes = []
    gt_classes = []
    pred_boxes = []
    pred_classes = []
    pred_scores = []

    for i, [image, boxes, classes, path] in tqdm(enumerate(eval_loader)):
        with torch.no_grad():
            if torch.cuda.is_available():
                image = image.cuda()

            prediction = model(image)

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

            frame = overlay_colour(prediction, frame, centroid)

            scores = get_score_from_prediction(prediction_softmax, coordinates)

            pred_boxes = convert_coordinates_to_bbox(coordinates)
            scores_new = []
            pred_boxes_new = []

            for box, score in zip(pred_boxes, scores):
                if score > args.conf_thresh:
                    pred_boxes_new.append(box)
                    scores_new.append(score)
            scores = scores_new

            pred_boxes = upsample_boxes(pred_boxes_new, prediction.shape, image.shape)

            pred_scores = np.asarray(scores)
            pred_classes = np.asarray([0] * len(coordinates))

            gt_boxes = boxes.numpy()
            gt_classes = classes.numpy()

            for box in pred_boxes:
                frame = cv2.rectangle(
                    frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2
                )
            for box in gt_boxes:
                frame = cv2.rectangle(
                    frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2
                )

            write_txt(
                (gt_boxes, gt_classes),
                (pred_boxes, pred_classes, pred_scores),
                decoder,
                path.split("/")[-1],
                args.output_path,
            )

            cv2.imwrite(
                os.path.join(args.output_path, "plotted_images", path.split("/")[-1]),
                frame,
            )


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
        "--output_path",
        type=str,
        required=False,
        default="./",
        help="Path to weights for which inference needs to be done",
    )

    args = parser.parse_args()

    model = hrnet(args.n_classes)

    if torch.cuda.is_available():
        model.cuda()

    model.load_state_dict(torch.load(args.weight_path)["state_dict"])
    model.eval()

    os.makedirs(os.path.join(args.output_path, "groundtruths"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "detections"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "plotted_images"), exist_ok=True)

    decoder = {k: 0 for k in range(10)}

    eval_dataset = YoloDataset(
        args.txt_path,
        train=False,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False, collate_fn=eval_dataset.collate_fn
    )

    validate_one_epoch(model, eval_loader, args, decoder)


if __name__ == "__main__":
    main()