import math
from torchvision import transforms
import torch


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