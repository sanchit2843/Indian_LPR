import math
from torchvision import transforms
import torch
import numpy as np


def sort_by_score(pred_boxes, pred_labels, pred_scores):
    score_seq = [(-score).argsort() for index, score in enumerate(pred_scores)]
    pred_boxes = [
        sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, score_seq)
    ]
    pred_labels = [
        sample_boxes[mask] for sample_boxes, mask in zip(pred_labels, score_seq)
    ]
    pred_scores = [
        sample_boxes[mask] for sample_boxes, mask in zip(pred_scores, score_seq)
    ]
    return pred_boxes, pred_labels, pred_scores


transformation = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


def preprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = transforms.Normalize(mean=mean, std=std)(transformation(image))
    return torch.unsqueeze(image, dim=0)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def postprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = UnNormalize(mean, std)(image)
    return (image * 255).squeeze(0).transpose(1, 2, 0).numpy().astype(np.uint8)


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


def lr_func(GLOBAL_STEPS, TOTAL_STEPS, WARMPUP_STEPS, LR_INIT, LR_END):
    if GLOBAL_STEPS < WARMPUP_STEPS:
        lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
    else:
        lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
            (
                1
                + math.cos(
                    (GLOBAL_STEPS - WARMPUP_STEPS)
                    / (TOTAL_STEPS - WARMPUP_STEPS)
                    * math.pi
                )
            )
        )
    return float(lr)