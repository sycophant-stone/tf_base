import os

def mkdir(dr):
    """
    Make dir if dir not exist.
    Args:
        dr: dir to make
    Return:
        None
    """
    if not os.path.exists(dr):
        os.makedirs(dr)

def list_all_files(dir_name, exts=["jpg", "bmp", "png"]):
    """
    List all file if ext in exts.
    Args: 
        dir_name: path to list
        exts: suffix list
    Return:
        a list contains all files that meet the requirements.
    """
    result = []
    for dir_, subdirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir_, file_name)) 
    return result

def IoU(box=None, gt_box=None):
    """
    Calculate the IoU value between two boxes.
    Args:
        box: box location information [xmin, ymin, xmax, ymax]
        gt_box: another box location information [xmin, ymin, xmax, ymax]
    Return:
        IoU Value, float.
    """
    area_box = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area_gt_box = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
    inter_x1 = max(box[0], gt_box[0])
    inter_y1 = max(box[1], gt_box[1])
    inter_x2 = min(box[2], gt_box[2])
    inter_y2 = min(box[3], gt_box[3])
    inter_w = max(0, inter_x2 - inter_x1 + 1)
    inter_h = max(0, inter_y2 - inter_y1 + 1)
    area_inter = inter_w * inter_h
    ovr = float(area_inter) / float(area_box + area_gt_box - area_inter)
    return ovr
