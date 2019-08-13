import os
import shutil
def fonders_for_pid(ptk_filename, img_path):
    pid_ptk_map ={}
    print('Read the file ... ')
    with open(filename, 'r') as f:
        for line in f.readlines():
            words = line.strip().split(',')
            pid = words[1]
            faceid = words[0]
            pidpath=img_path+"/"+pid
            print(pidpath)
            if os.path.exsits(pidpath):
                os.mkdir(pidpath)
            shutil.move(srcfile,dstfile)
            if pid not in pid_ptk_map:
                pid_ptk_map[pid] = set([])
            pid_ptk_map[pid].add(faceid)
    print('... done')
    
    

if __name__ == '__main__':
    #fonders_for_pid("ptk_pid_crossday_larger_top300.csv", img_path = "/ssd/hnren/3_K11_DATA/")
    fonders_for_pid("test_top2.csv", img_path = "/forp/")
