import os
from src.object_detection.model.fcos import FCOSDetector
from src.object_detection.model.config import DefaultConfig

from src.object_detection.utils.utils import preprocess_image

from torch import nn as nn
import argparse
import cv2
import torch

from src.Licence_Plate_Recognition.model.LPRNet import build_lprnet
from src.Licence_Plate_Recognition.test_LPRNet import Greedy_Decode_inference

import numpy as np
import json


def run_single_frame(od_model, lprnet, image):
    """[summary]

    Args:
        od_model ([type]): [description]
        lprnet ([type]): [description]
        image ([type]): [description]

    Returns:
        [type]: [description]
    """
    original_image = image.copy()
    image = preprocess_image(image)
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():
        out = od_model(image)
        scores, classes, boxes = out
        boxes = boxes[0].cpu().numpy().tolist()
        classes = classes[0].cpu().numpy().tolist()
        scores = scores[0].cpu().numpy().tolist()

    plate_images = []
    for b in boxes:
        plate_image = original_image[b[1] : b[3], b[0] : b[2], :]
        im = cv2.resize(plate_image, (94, 24)).astype("float32")
        im -= 127.5
        im *= 0.0078125
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        plate_images.append(im)
    plate_labels = Greedy_Decode_inference(lprnet, torch.stack(plate_images, 0))
    out_dict = {}

    for idx, (box, label) in enumerate(zip(boxes, plate_labels)):
        out_dict.update({idx: {"boxes": box, "label": label}})

    return out_dict


def plot_single_frame_from_out_dict(image, dict):
    for _, v in dict.items():
        box, label = v["boxes"], v["label"]
        image = cv2.rectangle(
            image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), thickness=1
        )
        # cv2.put text for adding label
    return image


def process_directory(args, od_model, lprnet):

    for i in os.listdir(args.source):

        if os.path.splitext(i)[1] in ["avi", "mp4"]:
            process_video(
                os.path.join(args.source, i), od_model, lprnet, args.output_path
            )

        if os.path.splitext(i)[1] in ["jpg", "png"]:
            image = cv2.imread(os.path.join(args.source, i))
            out_dict = run_single_frame(od_model, lprnet, args.source)
            plotted_image = plot_single_frame_from_out_dict(image, out_dict)

            cv2.imwrite(
                os.path.join(args.output_path, "plots", i),
                plotted_image,
            )

            with open(
                os.path.join(
                    args.output_path,
                    "jsons",
                    i.replace("jpg", "json").replace("png", "json"),
                ),
                "w",
            ) as outfile:
                json.dump({args.source.split("/")[-1]: out_dict}, outfile)

    return


def frame_extract(path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
        success, image = vidObj.read()
        if success:
            yield image


def process_video(video_path, od_model, lprnet, output_path):

    current_video = cv2.VideoCapture(video_path)
    fps = current_video.get(cv2.CAP_PROP_FPS)
    final_dict = {}

    for idx, frame in enumerate(frame_extract(video_path)):
        if idx == 0:
            out_video = cv2.VideoWriter(
                os.path.join(
                    args.output_dir, video_path.split("/")[-1].replace("mp4", "avi")
                ),
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                fps,
                (
                    frame.shape[0],
                    frame.shape[1],
                ),
            )
        out_dict = run_single_frame(od_model, lprnet, frame)
        out_frame = plot_single_frame_from_out_dict(frame, out_dict)
        final_dict.update({idx: out_dict})
        out_video.write(out_frame)

    out_video.release()
    with open(
        os.path.join(
            args.output_path,
            "jsons",
            video_path.split("/")[-1].replace("mp4", "json").replace("avi", "json"),
        ),
        "w",
    ) as outfile:
        json.dump(final_dict, outfile)
    return


def process_txt(args, od_model, lprnet):
    txt_file = open(args.source, "r")

    for i in txt_file.readlines():
        if os.path.splitext(i)[1] in ["avi", "mp4"]:
            process_video(
                os.path.join(args.source, i), od_model, lprnet, args.output_path
            )

        if os.path.splitext(i)[1] in ["jpg", "png"]:
            image = cv2.imread(os.path.join(args.source, i))
            out_dict = run_single_frame(od_model, lprnet, args.source)
            plotted_image = plot_single_frame_from_out_dict(image, out_dict)

            cv2.imwrite(
                os.path.join(args.output_path, "plots", i),
                plotted_image,
            )

            with open(
                os.path.join(
                    args.output_path,
                    "jsons",
                    i.replace("jpg", "json").replace("png", "json"),
                ),
                "w",
            ) as outfile:
                json.dump({args.source.split("/")[-1]: out_dict}, outfile)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # add more formats based on what is supported by opencv
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Location to image/folder/video/txt, image formats supported - jpg/png video formats supported - mp4,avi",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="output directory to save plotted images and text files with results",
    )

    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_path, "plots"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, "jsons"), exist_ok=True)

    # load object detection model

    od_model = FCOSDetector(mode="inference", config=DefaultConfig).eval()
    od_model.load_state_dict(
        torch.load(
            "weights/best_od.pth",
            map_location=torch.device("cpu"),
        )
    )

    # load ocr

    lprnet = build_lprnet(lpr_max_len=15, class_num=36).eval()
    lprnet.load_state_dict(
        torch.load("weights/best_lprnet.pth", map_location=torch.device("cpu"))
    )

    if torch.cuda.is_available():
        od_model = od_model.cuda()
        lprnet = lprnet.cuda()

    if os.path.isdir(args.source):
        process_directory(args, od_model, lprnet)

    else:
        if os.path.splitext(args.source)[1] in ["png", "jpg"]:

            image = cv2.imread(args.source)
            out_dict = run_single_frame(od_model, lprnet, args.source)
            plotted_image = plot_single_frame_from_out_dict(image, out_dict)

            cv2.imwrite(
                os.path.join(args.output_path, "plots", "plotted_image.png"),
                plotted_image,
            )

            with open(
                os.path.join(args.output_path, "jsons", "output.json"), "w"
            ) as outfile:
                json.dump({args.source.split("/")[-1]: out_dict}, outfile)

        if os.path.splitext(args.source)[1] in ["avi", "mp4"]:
            process_video(args.source, od_model, lprnet)

        if os.path.splitext(args.source)[1] == "txt":
            process_txt(args.source, od_model, lprnet)

    print("processing done")
