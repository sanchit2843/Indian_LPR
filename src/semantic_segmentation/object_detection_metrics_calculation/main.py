import os
from object_detection_metrics_calculation.src.coco_evaluator import (
    get_coco_summary,
)
from object_detection_metrics_calculation.src.bounding_box import (
    BoundingBox,
)
import argparse
from object_detection_metrics_calculation.src.enumerators import (
    CoordinatesType,
    BBType,
    BBFormat,
)
import pandas as pd
import math
import numpy as np
from tqdm import tqdm


def get_coco_metrics_from_path(path_to_results):
    all_gt_boxes = []
    all_detection_boxes = []
    each_image_metrics = []
    for i in tqdm(os.listdir(os.path.join(path_to_results, "groundtruths"))):
        gt_txt_file = open(os.path.join(path_to_results, "groundtruths", i), "r")
        detection_txt_file = open(os.path.join(path_to_results, "detections", i), "r")
        gt_boxes = []
        detected_boxes = []

        for current_line in gt_txt_file.readlines():
            current_line = current_line.split(" ")

            gt_boxes.append(
                BoundingBox(
                    image_name=i,
                    class_id=current_line[0],
                    coordinates=(
                        float(current_line[1]),
                        float(current_line[2]),
                        float(current_line[3]),
                        float(current_line[4]),
                    ),
                    type_coordinates=CoordinatesType.ABSOLUTE,
                    bb_type=BBType.GROUND_TRUTH,
                    confidence=None,
                    format=BBFormat.XYWH,
                )
            )
        for current_line in detection_txt_file.readlines():
            current_line = current_line.split(" ")

            detected_boxes.append(
                BoundingBox(
                    image_name=i,
                    class_id=current_line[0],
                    coordinates=(
                        float(current_line[2]),
                        float(current_line[3]),
                        float(current_line[4]),
                        float(current_line[5]),
                    ),
                    type_coordinates=CoordinatesType.ABSOLUTE,
                    bb_type=BBType.DETECTED,
                    confidence=float(current_line[1]),
                    format=BBFormat.XYX2Y2,
                )
            )
        all_gt_boxes += gt_boxes
        all_detection_boxes += detected_boxes

        image_metrics = get_coco_summary(gt_boxes, detected_boxes)
        image_metrics_list = [i]
        for _, v in image_metrics.items():
            if math.isnan(v):
                image_metrics_list.append(-1)
                continue
            image_metrics_list.append(v)
        each_image_metrics.append(np.array(image_metrics_list))

    all_image_metrics = get_coco_summary(all_gt_boxes, all_detection_boxes)

    return each_image_metrics, all_image_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_results",
        "-p",
        type=str,
        default="./example_result_folder",
        required=False,
        help="Path to result folder in structure defined in readme",
    )

    args = parser.parse_args()

    each_image_metrics, all_image_metrics = get_coco_metrics_from_path(
        args.path_to_results
    )
    image_metrics_list = ["all_images"]

    for _, v in all_image_metrics.items():
        if math.isnan(v):
            image_metrics_list.append(-1)
            continue
        image_metrics_list.append(v)

    each_image_metrics.append(np.array(image_metrics_list))
    each_image_metrics = pd.DataFrame(np.array(each_image_metrics))
    each_image_metrics.columns = ["image name"] + list(all_image_metrics.keys())

    each_image_metrics.to_csv(
        os.path.join(args.path_to_results, "each_image_results.csv"), index=False
    )

    print("object detection coco metrics for all images")
    print(all_image_metrics)
