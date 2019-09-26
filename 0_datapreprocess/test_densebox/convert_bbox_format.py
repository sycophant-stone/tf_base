import os, sys
import shutil
import random
import math
import json
import csv

def mkdir(dr):
    if not os.path.exists(dr):
        os.makedirs(dr)
        
def read_all(file_names):
    lines = []
    for file_name in file_names:
        with open(file_name,'rt') as f:
            lines.extend(f.readlines())
    return lines

# list all image files
def list_all_files(dir_name, exts = ['jpg', 'bmp', 'png']):
    result = []
    for dir, sub_dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir, file_name))
    return result

def convert_format():
    anno_file_path = "/data/xulifeng/head_for_label/20181114_head_label_part1.json"
    img_root = "/data/xulifeng/headcrop2/mall_bbox_2"
    with open(anno_file_path,'r') as f:
        data = json.load(f)
    for img_name in data.keys():
        bbox = data[img_name][0][0:4]
        base_name = os.path.splitext(img_name)[0]
        out_file = os.path.join(img_root, base_name + ".head.txt")
        print("write:", out_file)
        with open(out_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(bbox)
    
if __name__ == "__main__":
    convert_format()
    pass
