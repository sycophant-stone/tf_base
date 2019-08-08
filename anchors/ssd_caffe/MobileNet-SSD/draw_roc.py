import numpy as np
import os
import matplotlib.pyplot as plt


def draw_roc(dat_file):
    if not os.path.exists(dat_file):
        raise Exception("%s does not exist"%(dat_file))
    dat_map={}
    with open(dat_file,'r') as f:
        for line in f.readlines():
            if 'class1' in line:
                cls_val=line.split('class1: ')[1]
                continue
            words=line.split('=')
            rs=words[0].find('[')
            re=words[0].find('=')
            #print("words[0]:",words[0])
            #print("rs:%d,re:%d"%(rs+1,re-1))
            recall_id=words[0][rs+1:re]
            precision = words[1]
            dat_map[int(recall_id)]=precision
    dat_map_sorted = sorted(dat_map.items(), key=lambda x:x[0], reverse=True)
    print(type(dat_map_sorted))
    #dat_array = np.array(dat_map_sorted)
    #print(dat_array)
    dat_array = []
    for dms in dat_map_sorted:
        dat_array.append(float(dms[1]))
    #print(dat_map_sorted[0][0])
    print(dat_array)
    tick=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    plt.figure()
    #plt.subplot(2,2,1)
    plt.title('Cyclist, AP=%s'%(cls_val))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1.05])
    plt.xticks(tick)
    plt.yticks(tick)
    plt.plot(tick, dat_array)
    plt.savefig("p-r,png")













if __name__=='__main__':
    draw_roc(dat_file = 'p-r.dat')
