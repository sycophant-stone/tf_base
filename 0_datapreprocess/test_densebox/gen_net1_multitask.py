import os, sys
import shutil


def read_all(file_names):
    lines = []
    for file_name in file_names:
        with open(file_name,'rt') as f:
            lines.extend(f.readlines())
    return lines

def mkdir(dr):
    if not os.path.exists(dr):
        os.makedirs(dr)

# list all files
def list_all_files(dir_name, exts = ['jpg', 'bmp', 'png']):
    result = []
    for dir, sub_dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir, file_name))
    return result

def concat_files(in_filenames, out_filename):
    with open(out_filename, 'w') as outfile:
        for fname in in_filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
                    
def gen_positive_list(stage):
    workdir = "/ssd/xulifeng/workspace/xhd_stage1/train_data_" + stage
    positives = [os.path.join(workdir, "positives", "positive_lst.txt"),
                 os.path.join(workdir, "positives_add0", "positive_lst.txt"),
                 ]
    partfaces = [os.path.join(workdir, "positives", "partface_lst.txt"),
                 os.path.join(workdir, "positives_add0", "partface_lst.txt"),
                 ]

    concat_files(positives, os.path.join(workdir, "positive_lst.txt"))
    concat_files(partfaces, os.path.join(workdir, "partface_lst.txt"))    

def gen_negative_list(stage):
    workdir = "/ssd/xulifeng/workspace/xhd_stage1/train_data_" + stage
    negatives_imgset = os.path.join(workdir, "negatives")
    out_negative_file = os.path.join(workdir, "negative_lst.txt")
    
    lines = list_all_files(negatives_imgset, exts = ['jpg', 'bmp', 'png'])
    print "negatives = ", len(lines)
    
    f = open(out_negative_file, "wt")
    for line in lines:
        f.write(line + ",0,-100,-100,-100,-100" + "\n")
    f.close()

def line_append(line):
    line = line.strip() + ",-1,-100,-100,-100,-100"
    return line
    
def line_prepend(line):
    v = line.strip().split(",")
    line = v[0] + ",-1,-100,-100,-100,-100," + ",".join(v[1:])
    return line
    
def gen_trainset_tp1(out_dir):
    # positives
    workdir = "/ssd/xulifeng/workspace/xhd_stage1/train_data_net1"
    src_file_name = os.path.join(workdir, "positive_lst.txt")
    dst_file_name = os.path.join(out_dir, "tp1_positive_lst.txt")
    with open(dst_file_name, 'w') as outfile:
        for line in open(src_file_name, "r"):
            line = line_append(line)
            outfile.write(line + "\n")
            
    # part faces
    src_file_name = os.path.join(workdir, "partface_lst.txt")
    dst_file_name = os.path.join(out_dir, "tp1_partface_lst.txt")
    with open(dst_file_name, 'w') as outfile:
        for line in open(src_file_name, "r"):
            line = line_append(line)
            outfile.write(line + "\n")
            
    # negatives
    src_file_name = os.path.join(workdir, "negative_lst.txt")
    dst_file_name = os.path.join(out_dir, "tp1_negative_lst.txt")
    #shutil.copyfile(src_file_name, dst_file_name)
    
    with open(dst_file_name, 'w') as outfile:
        for line in open(src_file_name, "r"):
            line = line_append(line)
            outfile.write(line + "\n")
    
    
    
def gen_trainset_tp2(out_dir):
    # positives
    workdir = "/ssd/xulifeng/workspace/xhd_stage1/train_data_net2"
    src_file_name = os.path.join(workdir, "positive_lst.txt")
    dst_file_name = os.path.join(out_dir, "tp2_positive_lst.txt")
    with open(dst_file_name, 'w') as outfile:
        for line in open(src_file_name, "r"):
            line = line_prepend(line)
            outfile.write(line + "\n")
            
    # part faces
    src_file_name = os.path.join(workdir, "partface_lst.txt")
    dst_file_name = os.path.join(out_dir, "tp2_partface_lst.txt")
    with open(dst_file_name, 'w') as outfile:
        for line in open(src_file_name, "r"):
            line = line_prepend(line)
            outfile.write(line + "\n")
            
    # negatives
    src_file_name = os.path.join(workdir, "negative_lst.txt")
    dst_file_name = os.path.join(out_dir, "tp2_negative_lst.txt")
    #shutil.copyfile(src_file_name, dst_file_name)
    
    with open(dst_file_name, 'w') as outfile:
        for line in open(src_file_name, "r"):
            line = line_prepend(line)
            outfile.write(line + "\n")
            
if __name__ == '__main__':
    print("gen positive list")
    #gen_positive_list("net1")
    #gen_positive_list("net2")
    
    print("gen negative list")
    gen_negative_list("net1")
    gen_negative_list("net2")
    
    print("gen multi-task list")
    out_dir = "/ssd/xulifeng/workspace/xhd_stage1/train_data"
    mkdir(out_dir)
    gen_trainset_tp1(out_dir)
    gen_trainset_tp2(out_dir)
    
    pass
    
