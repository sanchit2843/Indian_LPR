import math
import numpy as np
import cv2
import os
from torchvision import transforms
import torch
from PIL import Image, ImageDraw
from scipy.spatial import distance
import matplotlib.pyplot as plt


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


def overlay_colour(frame, centroid):

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
        box = [[max(0, int(x[0])), max(0, int(x[1]))] for x in box]
        coordinates.append(box)

    return coordinates, centroid


import matplotlib.pyplot as plt


def get_warped_plates(rgb_image, coordinates):
    cropped_images = []

    for box in coordinates:

        height = int(distance.euclidean(box[0], box[1]))
        width = int(distance.euclidean(box[1], box[2]))

        src_pts = np.array(box).astype("float32")
        dst_pts = np.array(
            [[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]],
            dtype="float32",
        )

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        warped = cv2.warpPerspective(rgb_image, M, (width, height))

        if width < height:
            warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

        cropped_images.append(warped)

    return cropped_images


transformation = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


def preprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = transforms.Normalize(mean=mean, std=std)(transformation(image))
    return torch.unsqueeze(image, dim=0)


def postprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    inv_normalize = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / std[0], 1 / std[1], 1 / std[2]]
            ),
            transforms.Normalize(
                mean=[-1 * mean[0], -1 * mean[1], -1 * mean[2]], std=[1.0, 1.0, 1.0]
            ),
        ]
    )
    image = inv_normalize(image)
    return (
        (image * 255).squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
    )


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


def upsample_coordinates(coordinates, prediction_shape, image_shape):
    p_h, p_w = prediction_shape
    i_h, i_w, _ = image_shape
    coordinates_new = []
    boxes_new = []

    for c in coordinates:
        coordinates_new.append(
            [[int(x[0] * i_w / p_w), int(x[1] * i_h / p_h)] for x in c]
        )
    boxes_new = convert_coordinates_to_bbox(coordinates_new)
    return coordinates_new, boxes_new


def convert_x_y_tuple_to_xy_list(poly):
    x = []
    y = []
    for i in poly:
        x.append(i[0])
        y.append(i[1])
    return (x, y)


def convert_polylist_to_tuple(poly):
    return [(x[0], x[1]) for x in poly]


def get_score_and_class_from_prediction(prediction, prediction_softmax, coordinates):
    scores = []
    classes = []

    mask = np.zeros(
        (
            prediction_softmax.shape[2],
            prediction_softmax.shape[3],
        ),
        dtype=np.uint8,
    )

    prediction_desired_class = prediction_softmax[0, 0, :, :].cpu().numpy()

    for poly in coordinates:
        mask = Image.fromarray(mask)
        draw = ImageDraw.Draw(mask)
        draw.polygon(xy=convert_polylist_to_tuple(poly), outline=1, fill=1)
        mask = np.asarray(mask)

        value, count = np.unique(prediction * mask, return_counts=True)
        # remove 0 from value and count
        value = list(value)
        count = list(count)
        del count[value.index(0)]
        value.remove(0)

        desired_class = value[np.argmax(count)]
        classes.append(desired_class)
        scores.append(
            np.sum(prediction_softmax[0, desired_class, :, :].cpu().numpy() * mask)
            / np.sum(mask)
        )

    return classes, scores
