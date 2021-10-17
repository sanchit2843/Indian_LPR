import os
from src.semantic_segmentation.models.hrnet import hrnet
from src.semantic_segmentation.utils.util import (
    get_warped_plates,
    plate_locate,
    get_score_and_class_from_prediction,
    preprocess_image,
    upsample_coordinates,
    convert_coordinates_to_bbox,
)

from torch import nn as nn
import argparse
import cv2
import torch

from src.License_Plate_Recognition.model.LPRNet import build_lprnet
from src.License_Plate_Recognition.test_LPRNet import Greedy_Decode_inference

import numpy as np
import json
from tqdm import tqdm


def run_single_frame(semantic_model, lprnet, image, conf_thresh):
    """[summary]

    Args:
        semantic_model ([type]): [description]
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
        out = semantic_model(image, (image.shape[2], image.shape[3]))
        prediction_softmax = nn.Softmax(dim=1)(out["output"])
        out = (
            torch.argmax(out["output"], dim=1)
            .detach()
            .cpu()
            .squeeze(dim=0)
            .numpy()
            .astype(np.uint8)
        )
        coordinates, _ = plate_locate(out)
        scores = get_score_and_class_from_prediction(out,prediction_softmax, coordinates)
        pred_boxes = convert_coordinates_to_bbox(coordinates)

        pred_boxes_new = []
        coordinates_new = []
        for box, score, c in zip(pred_boxes, scores, coordinates):

            if score[0] > args.conf_thresh:
                pred_boxes_new.append(box)
                coordinates_new.append(c)

        coordinates, boxes = upsample_coordinates(
            coordinates_new, out.shape, original_image.shape
        )

    if len(boxes) == 0:
        return {0: {"coordinates": [], "boxes": [], "label": ""}}

    plate_images = get_warped_plates(original_image, coordinates)
    plate_images_tensor = []
    for plate_image in plate_images:
        im = cv2.resize(plate_image, (94, 24)).astype("float32")
        im -= 127.5
        im *= 0.0078125
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        plate_images_tensor.append(im)

    plate_labels = Greedy_Decode_inference(lprnet, torch.stack(plate_images_tensor, 0))
    out_dict = {}

    for idx, (box_4pnt, box, label) in enumerate(zip(coordinates, boxes, plate_labels)):
        out_dict.update({idx: {"coordinates": box_4pnt, "boxes": box, "label": label}})

    return out_dict


def plot_single_frame_from_out_dict(im, out_dict,line_thickness=3,color = (255,0,0)):
    if out_dict:
      for _, v in out_dict.items():
        box, label = v["boxes"], v["label"]
        
        if len(box) < 4:
            continue

        tl = (
            line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
        )  # line/font thickness
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        mask = im.copy()
        mask[int(box[1]):int(box[3]),int(box[0]):int(box[2])] = color
        im = cv2.addWeighted(im, 0.7, mask, 0.3, 0)
        # cv2.rectangle(im, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                im,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [225, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
    return im


def process_directory(args, semantic_model, lprnet):

    for i in tqdm(os.listdir(args.source)):

        if os.path.splitext(i)[1] in [".avi", ".mp4"]:
            process_video(
                os.path.join(args.source, i), semantic_model, lprnet, args.output_path
            )

        if os.path.splitext(i)[1] in [".png", ".jpg"]:
            image = cv2.imread(os.path.join(args.source, i))
            out_dict = run_single_frame(semantic_model, lprnet, image, args.conf_thresh)

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


def process_video(video_path, semantic_model, lprnet, output_dir):

    current_video = cv2.VideoCapture(video_path)
    fps = current_video.get(cv2.CAP_PROP_FPS)
    final_dict = {}
    print('processing {}'.format(video_path))
    for idx, frame in enumerate(frame_extract(video_path)):
        if idx == 0:
            out_video = cv2.VideoWriter(
                os.path.join(
                    output_dir, video_path.split("/")[-1].replace("mp4", "avi")
                ),
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                fps,
                (
                    frame.shape[1],
                    frame.shape[0],
                ),
            )
        out_dict = run_single_frame(semantic_model, lprnet, frame, args.conf_thresh)
        out_frame = plot_single_frame_from_out_dict(frame, out_dict)
        final_dict.update({idx: out_dict})
        out_video.write(out_frame)

    out_video.release()
    with open(
        os.path.join(
            output_dir,
            "jsons",
            video_path.split("/")[-1].replace("mp4", "json").replace("avi", "json"),
        ),
        "w",
    ) as outfile:
        json.dump(final_dict, outfile)
    return


def process_txt(args, semantic_model, lprnet):
    txt_file = open(args.source, "r")

    for i in txt_file.readlines():
        if os.path.splitext(i)[1] in [".avi", ".mp4"]:
            process_video(
                os.path.join(args.source, i), semantic_model, lprnet, args.output_path
            )

        if os.path.splitext(i)[1] in [".jpg", ".png"]:
            image = cv2.imread(os.path.join(args.source, i))

            out_dict = run_single_frame(semantic_model, lprnet, image, args.conf_thresh)

            if out_dict:
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
        "--conf_thresh",
        type=int,
        default=0.5,
        help="output directory to save plotted images and text files with results",
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
    if torch.cuda.is_available():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group("gloo", rank=0, world_size=1)

    semantic_model = hrnet().eval()
    if torch.cuda.is_available():
        semantic_model = nn.SyncBatchNorm.convert_sync_batchnorm(semantic_model)
    semantic_model = nn.DataParallel(semantic_model)
    semantic_model.load_state_dict(
        torch.load(
            "weights/best_semantic.pth",
            map_location=torch.device("cpu"),
        )["state_dict"]
    )

    # load ocr

    lprnet = build_lprnet(lpr_max_len=16, class_num=37).eval()
    lprnet.load_state_dict(
        torch.load("weights/best_lprnet.pth", map_location=torch.device("cpu"))
    )

    if torch.cuda.is_available():
        semantic_model = semantic_model.cuda()
        lprnet = lprnet.cuda()

    if os.path.isdir(args.source):
        print("source is directory, might need time to process")
        process_directory(args, semantic_model, lprnet)

    else:

        if os.path.splitext(args.source)[1] in [".png", ".jpg"]:
            print("source is image")
            image = cv2.imread(args.source)
            out_dict = run_single_frame(semantic_model, lprnet, image, args.conf_thresh)
            if out_dict:
                plotted_image = plot_single_frame_from_out_dict(image, out_dict)

                cv2.imwrite(
                    os.path.join(args.output_path, "plots", "plotted_image.png"),
                    plotted_image,
                )

                with open(
                    os.path.join(args.output_path, "jsons", "output.json"), "w"
                ) as outfile:
                    json.dump({args.source.split("/")[-1]: out_dict}, outfile)

        if os.path.splitext(args.source)[1] in [".avi", ".mp4"]:
            print("source is video")
            process_video(args.source, semantic_model, lprnet,args.output_path)

        if os.path.splitext(args.source)[1] == ".txt":
            print("source is txt, might need time to process")
            process_txt(args, semantic_model, lprnet)

    print("processing done")
