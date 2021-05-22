import os


def write_txt(gt, pred, decoder, image_name, output_path):
    """[This code will can be used to write detected and ground truth metrics in the format used in object detection metrics calculation pipeline]

    Text file format is:
    ground truth: <class> <left> <top> <width> <height>
    detection:  <class> <confidence> <left> <top> <right> <bottom>

    Args:
        gt ([tuple]): (ground truth boxes,ground truth classes)
        pred ([tuple]): (detected boxes, predicted classes, predicted scores)
        decoder ([dict]): a dictionary mapping class id with class name.
        image_name ([str]): name of image
        output_path ([str]): path to output folder, two sub folders groundtruths and detections will be created inside this folder
    """

    gt_boxes, gt_classes = gt
    # gt boxes format: [[x1,y1,x2,y2]] a list of bounding box coordinates in format <left> <top> <right> <bottom>
    # gt classes format: A list of predicted classes(integer) for each box. [1,3,0,2]

    pred_boxes, pred_classes, pred_scores = pred
    # pred boxes format: [[x1,y1,x2,y2]] a list of bounding box coordinates in format <left> <top> <right> <bottom>
    # pred classes format: A list of predicted classes(integer) for each box. [1,3,0,2]
    # pred scores format: A list of predicted confidence scores(float) for each box. [0.8,0.6,0.9]

    f_gt = open(
        os.path.join(output_path, "groundtruths/{}.txt".format(image_name)), "w+"
    )
    f_pred = open(
        os.path.join(output_path, "detections/{}.txt".format(image_name)), "w+"
    )

    for b, c in zip(gt_boxes, gt_classes):

        f_gt.write(
            "{} {} {} {} {}\n".format(decoder[c], b[0], b[1], b[2] - b[0], b[3] - b[1])
        )

    for b, c, s in zip(pred_boxes, pred_classes, pred_scores):
        f_pred.write(
            "{} {} {} {} {} {}\n".format(decoder[c], s, b[0], b[1], b[2], b[3])
        )