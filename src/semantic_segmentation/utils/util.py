import math
import numpy as np
import cv2
import os
from torchvision import transforms
import torch
from PIL import Image, ImageDraw


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        """
        Average meter constructor

        """
        self.reset()

    def reset(self):
        """
        Reset AverageMeter

        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update value

        Parameters:
            val (float): Value to add in the meter
            n (int): Batch size

        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


classes = {
    1: [(255, 0, 0), "plate"],
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


def plate_locate(image, size_factor=1, area_thresh=600):

    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    coordinates = []
    centroid = []

    for c in cnts:
        area = cv2.contourArea(c)

        if area < area_thresh:
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


transformation = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


def preprocess_image(image, mean, std):
    image = transforms.Normalize(mean=mean, std=std)(transformation(image))
    return torch.unsqueeze(image, dim=0)


def convert_yolotxtline_to_bboxes(current_line):
    boxes = []
    classes = []
    path = current_line.split(" ")[0]
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


def upsample_boxes(boxes, prediction_shape, image_shape):
    p_h, p_w = prediction_shape
    i_h, i_w = image_shape
    boxes_new = []
    for b in boxes:
        x1 = int(b[0] * i_w / p_w)
        y1 = int(b[1] * i_h / p_h)
        x2 = int(b[2] * i_w / p_w)
        y2 = int(b[3] * i_h / p_h)
        boxes_new.append([x1, y1, x2, y2])
    return boxes_new


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