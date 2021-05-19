import torch
import numpy as np
import cv2
import pandas as pd
import os


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
            "{} {} {} {} {} {}\n".format(
                decoder[c], s, int(b[0]), int(b[1]), int(b[2]), int(b[3])
            )
        )


if __name__ == "__main__":
    os.makedirs("predictions_for_eval", exist_ok=True)
    from model.fcos import FCOSDetector
    from demo import convertSyncBNtoBN
    from dataloader.custom_dataset import YoloDataset

    decoder = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
    }

    eval_dataset = YoloDataset(
        "/media/sanchit/Workspace/Projects/Paper_ANPR_dataset/Indian_Number_plate_recognition/Semantic_evaluate_map/test_yolo_public.txt",
        train=False,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=False, collate_fn=eval_dataset.collate_fn
    )
    from torch import nn

    model = FCOSDetector(mode="inference")

    model = nn.DataParallel(model)
    model.load_state_dict(
        torch.load(
            "/media/sanchit/Workspace/Projects/Indian_LPR/training/object_detection/od_weights/hrnet18v2_ranger48.pth",
            map_location=torch.device("cpu"),
        )
    )

    model = model.cuda().eval()
    print("===>success loading model")

    gt_boxes = []
    gt_classes = []
    pred_boxes = []
    pred_classes = []
    pred_scores = []
    num = 0
    from torchvision import transforms

    inv_normalize = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    data_ap = {}
    data_ap = np.zeros((len(eval_loader), 2), dtype=object)
    os.makedirs("./result_evaluation/plotted_images", exist_ok=True)
    os.makedirs("./result_evaluation/groundtruths", exist_ok=True)
    os.makedirs("./result_evaluation/detections", exist_ok=True)

    for i, [img, boxes, classes, paths] in tqdm(enumerate(eval_loader)):

        frame = (inv_normalize(img[0]) * 255).permute(1, 2, 0).numpy().astype(np.uint8)
        with torch.no_grad():
            out = model(img.cuda())
            pred_boxes.append(out[2][0].cpu().numpy())
            pred_classes.append(out[1][0].cpu().numpy())
            pred_scores.append(out[0][0].cpu().numpy())
        gt_boxes.append(boxes[0].numpy())
        gt_classes.append(classes[0].numpy())

        frame = frame.copy()
        for bbox in boxes[0].numpy():
            frame = cv2.rectangle(
                frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color=(0, 0, 255),
                thickness=4,
            )
        for bbox in out[2][0].cpu().numpy():
            frame = cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color=(255, 255, 0),
                thickness=3,
            )
        cv2.imwrite(
            os.path.join("./result_evaluation/plotted_images", paths[0].split("/")[-1]),
            frame,
        )

        pred_boxes, pred_classes, pred_scores = sort_by_score(
            pred_boxes, pred_classes, pred_scores
        )

        write_txt(
            (boxes[0].numpy(), classes[0].numpy()),
            (out[2][0].cpu().numpy(), out[1][0].cpu().numpy(), out[0][0].cpu().numpy()),
            decoder,
            paths[0].split("/")[-1],
        )
        pred_boxes = []
        pred_scores = []
        pred_classes = []
        gt_boxes = []
        gt_classes = []
