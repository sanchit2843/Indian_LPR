import numpy as np
import torch
from torch import nn
import argparse
import time
import os
import sys
import cv2
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import get_cfg_defaults
from models.model import create_model
from mask_to_bbox import create_sub_masks, mask_to_rectangle
import json
from tqdm import tqdm
import random
import math
from PIL import Image, ImageDraw
import shutil

transformation = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
classes = {
    10: [(102, 102, 0), "auto front"],
    9: [(104, 104, 104), "auto back"],
    8: [(255, 102, 102), "bus front"],
    7: [(255, 255, 0), "bus back"],
    5: [(255, 0, 127), "truck back"],
    6: [(204, 0, 204), "truck front"],
    4: [(102, 204, 0), "bike front"],
    2: [(0, 0, 255), "car front"],
    3: [(0, 255, 0), "bike back"],
    1: [(255, 0, 0), "car back"],
    0: [(0, 0, 0), "background"],
}


def overlay_colour(prediction, frame, centroid):

    temp_img = frame.copy()
    for i in range(len(centroid)):
        temp = centroid[i]
        box = cv2.boxPoints(temp)
        box = np.int0(box)
        cv2.drawContours(temp_img, [box], 0, classes[1][0], -1)
    cv2.addWeighted(temp_img, 0.5, frame, 0.5, 0, frame)
    return frame


def plate_locate(image):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    size_factor = 1.0  # hard code
    cropped_images = []
    coordinates = []
    centroid = []
    plate_areas = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 600:  # hard code
            continue

        temp_rect = []
        rect = cv2.minAreaRect(c)
        centroid.append(rect)
        temp_rect.append(rect[0][0])
        temp_rect.append(rect[0][1])
        temp_rect.append(rect[1][0] * size_factor)
        temp_rect.append(rect[1][1] * size_factor)
        temp_rect.append(rect[2])
        rect = (
            (temp_rect[0], temp_rect[1]),
            (temp_rect[2], temp_rect[3]),
            temp_rect[4],
        )

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = [[max(0, x[0]), max(0, x[1])] for x in box]
        coordinates.append(box)

    return coordinates, centroid


def normalize(image, cfg):
    return transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std)(
        transformation(image)
    )


def preprocess_image(image, cfg):
    image = normalize(image, cfg)
    return torch.unsqueeze(image, dim=0)


def convert_yolotxtline_to_bboxes(current_line):
    boxes = []
    classes = []
    path = current_line.split(" ")[0]
    try:
        if len(current_line.split(" ")[1:][0]) > 6:
            for bbox in current_line.split(" ")[1:]:
                a = []
                for idx, x in enumerate(bbox.split(",")):
                    if idx == 4:
                        classes.append(0)
                        break
                    if idx % 2 == 0:
                        a.append(max(0, int(int(x))))
                    else:
                        a.append(max(0, int(int(x))))
                boxes.append(a)
    except:
        print(path)
    return path, np.array(boxes), np.array(classes)


def convert_poly_to_bbox(x, y):
    x1 = min(x)
    x2 = max(x)
    y1 = min(y)
    y2 = max(y)
    bbox = [x1, y1, x2, y2]
    return bbox


def convert_coordinates_to_bbox(coordinates):
    boxes = []
    for i in coordinates:
        x, y = convert_x_y_tuple_to_xy_list(i)
        boxes.append(convert_poly_to_bbox(x, y))
    return boxes


def convert_x_y_tuple_to_xy_list(poly):
    x = []
    y = []
    for i in poly:
        x.append(i[0])
        y.append(i[1])
    return (x, y)


def convert_polylist_to_tuple(poly):
    return [(x[0], x[1]) for x in poly]


def get_score_from_prediction(prediction_softmax, coordinates):
    scores = []
    mask = np.zeros(
        (
            prediction_softmax.shape[2],
            prediction_softmax.shape[3],
        ),
        dtype=np.uint8,
    )
    prediction_desired_class = prediction_softmax[0, 1, :, :].cpu().numpy()

    for poly in coordinates:
        mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask)
        draw.polygon(xy=convert_polylist_to_tuple(poly), outline=1, fill=1)
        mask = np.asarray(mask)
        scores.append(np.sum(prediction_desired_class * mask) / np.sum(mask))

    return scores


def write_txt(gt, pred, decoder, image_name):
    """[output txt format]
    gt: <class> <left> <top> <width> <height>
    pred:  <class> <confidence> <left> <top> <right> <bottom>
    Args:
        gt ([type]): [description]
        pred ([type]): [description]
        decoder ([type]): [description]
        image_name ([type]): [description]
    """
    gt_boxes, gt_classes = gt
    pred_boxes, pred_classes, pred_scores = pred

    f_gt = open("./result_evaluation/groundtruths/{}.txt".format(image_name), "w+")
    f_pred = open("./result_evaluation/detections/{}.txt".format(image_name), "w+")
    for b, c in zip(gt_boxes, gt_classes):
        f_gt.write(
            "{} {} {} {} {}\n".format(decoder[c], b[0], b[1], b[2] - b[0], b[3] - b[1])
        )

    for b, c, s in zip(pred_boxes, pred_classes, pred_scores):

        f_pred.write(
            "{} {} {} {} {} {}\n".format(decoder[c], s, b[0], b[1], b[2], b[3])
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        type=str,
        default="",
        required=True,
        help="Location of current config file",
    )

    parser.add_argument(
        "--path_to_videos",
        type=str,
        required=True,
        default="./",
        help="Path where all the test videos are located",
    )

    parser.add_argument(
        "--path_to_weights",
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

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(["train.config_path", args.cfg])

    model = create_model(cfg)

    if torch.cuda.is_available():
        model.cuda()

    model.load_state_dict(torch.load(args.path_to_weights)["state_dict"])
    model.eval()

    gt_boxes = []
    gt_classes = []
    pred_boxes = []
    pred_classes = []
    pred_scores = []

    if os.path.exists("./result_evaluation"):
        shutil.rmtree("./result_evaluation")
    os.makedirs("./result_evaluation/groundtruths", exist_ok=True)
    os.makedirs("./result_evaluation/detections", exist_ok=True)
    os.makedirs("./result_evaluation/original_images", exist_ok=True)

    text_file = open(args.path_to_videos, mode="r+")
    list_sample = text_file.readlines()
    decoder = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}

    for idx in tqdm(range(len(list_sample))):
        current_line = list_sample[idx]
        path, boxes, classes = convert_yolotxtline_to_bboxes(current_line)
        frame = cv2.imread(path)
        image = preprocess_image(frame, cfg)

        boxes = torch.from_numpy(boxes)
        classes = torch.from_numpy(classes)

        with torch.no_grad():

            if torch.cuda.is_available():
                image = image.cuda()
            prediction = model(image, (frame.shape[0], frame.shape[1]))

            prediction_softmax = nn.Softmax(dim=1)(prediction["output"][0])

            prediction = (
                torch.argmax(prediction["output"][0], dim=1)
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
            pred_boxes = pred_boxes_new

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
            )
            cv2.imwrite(
                os.path.join(
                    "./result_evaluation/original_images", path.split("/")[-1]
                ),
                frame,
            )


if __name__ == "__main__":
    main()