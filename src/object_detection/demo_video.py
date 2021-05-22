# test/inference script
import numpy as np
import torch
from torch import nn
import argparse
import time
import os
import sys
import cv2
from torchvision import models
from tqdm import tqdm
from model.fcos import FCOSDetector
from torchvision import transforms
import matplotlib.pyplot as plt


def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image


transformation = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


def normalize(image):
    return transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )(transformation(image))


def preprocess_image(image):
    image = normalize(image)
    return torch.unsqueeze(image, dim=0)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        default="./",
        help="Path where all the test images are located or you can give path to video, it will break into each frame and write as a video",
    )

    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        default="./",
        help="Path to weights for which inference needs to be done",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="path to output saved video",
    )

    class Config:
        # backbone
        pretrained = True
        freeze_stage_1 = True
        freeze_bn = True

        # fpn
        fpn_out_channels = 256
        use_p5 = False

        # head
        class_num = 2
        use_GN_head = True
        prior = 0.01
        add_centerness = True
        cnt_on_reg = True

        strides = [4, 8, 16, 32, 64, 128]
        limit_range = [
            [-1, 32],
            [32, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, 999999],
        ]
        # inference
        score_threshold = 0.3
        nms_iou_threshold = 0.2
        max_detection_boxes_num = 150

    args = parser.parse_args()

    model = FCOSDetector(mode="inference", config=Config)
    model.load_state_dict(
        torch.load(
            args.path_to_weights,
            map_location=torch.device("cpu"),
        )
    )

    model = model.eval().cuda()

    current_video = cv2.VideoCapture(args.path_to_images)
    fps = current_video.get(cv2.CAP_PROP_FPS)

    for idx, frame in enumerate(tqdm(frame_extract(args.path_to_images))):
        if idx == 0:
            out_video = cv2.VideoWriter(
                os.path.join(
                    args.output_dir, args.path_to_images.split("/")[-1][:-3] + "avi"
                ),
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                fps,
                (
                    frame.shape[0],
                    frame.shape[1],
                ),
            )

        image = preprocess_image(frame)

        with torch.no_grad():
            out = model(image.cuda())
            scores, classes, boxes = out
            boxes = boxes[0].cpu().numpy().tolist()
            classes = classes[0].cpu().numpy().tolist()
            scores = scores[0].cpu().numpy().tolist()

            for box in boxes:
                pt1 = (int(box[0]), int(box[1]))
                pt2 = (int(box[2]), int(box[3]))
                frame = cv2.rectangle(frame, pt1, pt2, (255, 0, 0), thickness=3)
            out_video.write(frame)
        out_video.release()


if __name__ == "__main__":
    main()