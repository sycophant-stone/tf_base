from sklearn.metrics import roc_curve
from general_tools import IoU

def detection_result_match(gt_result=None, dt_result=None, IoU_thresh=None):
    """
    Match detection result and ground truth for roc curve draw.
    Args:
        gt_result: a dict {label0:{file1:[[xmin, ymin, xmax, ymax, ignore], ...], ...}, ...}
        dt_result: a dict {label0:{file1:[[xmin, ymin, xmax, ymax, score], ...], ...}, ...}
    Return:
        result_dict: a dict {label0:{"labels":labels, "scores":scores, "img_counter":img_counter, 
            "gt_counter":gt_counter, "dt_counter":dt_counter, ...}, ...}
    """
    if (gt_result is None) or (dt_result is None) or (IoU_thresh is None):
        raise ValueError("gt_result dt_result and IoU_thresh are required.")
    result_dict = {}
    for label in gt_result.keys():
        if len(gt_result[label]) != len(dt_result[label]):
            raise ValueError("Number of images in ground truth result != detection result.")
        img_counter = len(gt_result[label])
        dt_counter = 0
        gt_counter = 0
        labels = []
        scores = []
        for image_name in gt_result[label].keys():
            gt_boxes = gt_result[label][image_name]
            dt_boxes = dt_result[label][image_name]
            dt_counter += len(dt_boxes)
            gt_scores = [-1] * len(gt_boxes)
            dt_maps = [-1] * len(dt_boxes)
            for gt_idx, g_box in enumerate(gt_boxes):
                iou_max = 0
                iou_max_pos = -1
                for dt_idx, d_box in enumerate(dt_boxes):
                    print("d_box:",d_box)
                    print("g_box:",g_box)
                    iou = IoU(d_box, g_box)
                    print("iou:",iou)
                    if iou > iou_max:
                        iou_max = iou
                        iou_max_pos = dt_idx
                if iou_max > IoU_thresh:
                    gt_scores[gt_idx] = dt_boxes[iou_max_pos][4]
                    dt_maps[iou_max_pos] = gt_idx
            for gt_idx, gt_box in enumerate(gt_boxes):
                ignore = gt_box[4]
                if not ignore:
                    gt_counter += 1
                    labels.append(1)
                    scores.append(gt_scores[gt_idx])
            for dt_idx, dt_box in enumerate(dt_boxes):
                if dt_maps[dt_idx] < 0:
                    labels.append(0)
                    scores.append(dt_box[4])
        result_dict[label] = {"labels":labels, 
                "scores":scores, "img_counter":img_counter, 
                "gt_counter":gt_counter, "dt_counter":dt_counter}
    return result_dict

def get_recall_and_thresh(fppi, tpr, thresholds, value=0.25):
    """
    Get recall rate value according to fppi value.
    Args:
        fppis, recalls, thresholds
    Returns:
        fppi, tpr, thresholds according to the fppi value.
    """
    abs_distance = abs(fppi[0] - value)
    idx = 0
    for i in range(1, len(fppi)):
        dist = abs(fppi[i] - value)
        if dist < abs_distance:
            abs_distance = dist
            idx = i
    return fppi[idx], tpr[idx], thresholds[idx]

def calculate_fppi(match_result=None, fppi_values=None):
    """
    Calculate the recall threshold for different fppi values.
    Args:
        match_result: The result of the match between the test result and ground truth.
        fppi_values: specified fppi value.
    Returns:
        fppi, recall, thresh according to different fppi value.
    """
    fppis_result = dict()
    for label in match_result.keys():
        fp_counter = match_result[label]["labels"].count(0)
        img_counter = match_result[label]["img_counter"]
        fpr, tpr, thresholds = roc_curve(match_result[label]["labels"], 
                match_result[label]["scores"])
        N = len(fpr)
        fppi = [x*fp_counter/img_counter for x in fpr]
        fppi[-1] = 10000
        fppis_result[label] = {"fppi":[], "recall":[], "thresh":[]}
        for fppi_value in fppi_values:
            fppi_, recall_, thresh_ = get_recall_and_thresh(fppi, tpr, thresholds, value=fppi_value)
            fppis_result[label]["fppi"].append(fppi_)
            fppis_result[label]["recall"].append(recall_)
            fppis_result[label]["thresh"].append(thresh_)
    return fppis_result



