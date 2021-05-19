import numpy as np
import torch


def get_metrics_values(imLab, imPred, numClass):
    """
    Function used to get intersection and union value

    Parameters:

        imLab (torch.Tensor): Ground truth label
        imPred (torch.Tensor): Prediction of the model
        num_class (int): number of classes

    Returns:
        float: Intersection value
        float: Union value

    """

    imLab = imLab.detach().cpu().numpy()
    imPred = torch.argmax(imPred[0], dim=1).detach().cpu().numpy()
    inte, uni = intersectionAndUnion(imLab, imPred, numClass)
    return inte + 1e-9, uni + 1e-9


def intersectionAndUnion(label, seg_pred, num_class, ignore=255):
    """
    Function used to calculate Intersection over Union (IoU)

    Parameters:

        label (np.ndarray): Ground truth label
        seg_pred (np.ndarray): Prediction of the model
        num_class (int): number of classes
        ignore (int): value of the pixel to ignore

    Returns:
        float: Intersection over Union (IoU) score

    """

    size = (label.shape[1], label.shape[2])
    #    seg_pred = pred.transpose(0, 2, 3, 1)
    #    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(label[:, : size[-2], : size[-1]], dtype=np.int)
    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype("int32")
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)
    return (tp, np.maximum(1.0, pos + res - tp))
