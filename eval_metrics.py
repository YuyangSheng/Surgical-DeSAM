import numpy as np

def compute_overlap(boxes, query_boxes):
    """
    Args
        boxes:       (N, 4) ndarray of float
        query_boxes: (4)    ndarray of float
    Returns
        overlaps: (N) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    overlaps = np.zeros((N), dtype=np.float64)
    box_area = (
        (query_boxes[2] - query_boxes[0]) *
        (query_boxes[3] - query_boxes[1])
    )
    for n in range(N):
        iw = (
            min(boxes[n, 2], query_boxes[2]) -
            max(boxes[n, 0], query_boxes[0])
        )
        if iw > 0:
            ih = (
                min(boxes[n, 3], query_boxes[3]) -
                max(boxes[n, 1], query_boxes[1])
            )
            if ih > 0:
                ua = np.float64(
                    (boxes[n, 2] - boxes[n, 0]) *
                    (boxes[n, 3] - boxes[n, 1]) +
                    box_area - iw * ih
                )
                overlaps[n] = iw * ih / ua
    return overlaps

def check_if_true_or_false_positive(gt_boxes, pred_boxes, probas, iou_threshold=0.3):
    '''
    Args:
    gt_boxes: Ground-truth boxes with shape (N, 4)
    pred_boxes: Prediction bounding boxes with shape (N, 4)
    probas: Probabilities of predicted boxes with shape (N, 8)

    '''
    gt_boxes = np.array(gt_boxes, dtype=np.float64)
    pred_scores = np.max(probas, axis=1) # (N, 1)
    labels = np.argmax(probas, axis=1) 
    sorted_indices = np.argmax(pred_scores)

    # sort scores and pred_boxes in descending order
    pred_scores = pred_scores[sorted_indices]
    pred_boxes = pred_boxes[sorted_indices]

    N = pred_boxes.shape[0]

    scores = []
    false_positives = []
    true_positives = []
    detected_gt_idx = [] # a GT box should be mapped only one predicted box at most.
    
    for i in range(N):
        scores.append(pred_scores[i])

        if len(gt_boxes) == 0:
            false_positives.append(1)
            true_positives.append(0)
            continue

        overlaps = compute_overlap(gt_boxes, pred_boxes[i])
        assigned_gt_idx = np.argmax(overlaps)
        max_overlap = overlaps[assigned_gt_idx]

        if max_overlap >= iou_threshold and assigned_gt_idx not in detected_gt_idx:
            false_positives.append(0)
            true_positives.append(1)

            detected_gt_idx.append(assigned_gt_idx)
        else:
            false_positives.append(1)
            true_positives.append(0)

    return scores, false_positives, true_positives

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def mean_ap_for_boxes(gt_target, pred_boxes, iou_threshold=0.3):
    labels = gt_target['labels']
    unique_labels = np.unique(labels)

    for zz, label in enumerate(sorted(labels)):
        false_positives = []
        true_positives = []
        scores = []
        num_annotations = 0.0