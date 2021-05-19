import math


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


def lr_func(GLOBAL_STEPS, WARMPUP_STEPS, TOTAL_STEPS, LR_INIT, LR_END):
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