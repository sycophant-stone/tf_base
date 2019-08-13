import operator
import os 

def topk(filename, k=2):
    #ptk_pid_map = {}
    list_pid = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            words = line.strip().split(',')
            list_pid.append(words[1])
    #print("list_pid:", list_pid)
    dict_cnt = {}
    for i in list_pid:
        if list_pid.count(i)>=1:
            dict_cnt[i] = list_pid.count(i)
    #print("dict_cnt", dict_cnt)
    list_cnt_sorted = sorted(dict_cnt.items(), key=operator.itemgetter(1), reverse = True)
    #print("list_cnt_sorted", list_cnt_sorted)
    list_cnt_topk = list_cnt_sorted[:k]
    dict_cnt_sorted = {}
    for l in list_cnt_topk:
        #l0 pid, l1 cnt
        with open(filename, 'r') as f:
            for line in f.readlines():
                if l[0] in line:
                    fid = line.strip().split(',')[0]
                    dict_cnt_sorted[fid] = "%d_"%(list_cnt_topk.index(l)+1)+l[0]
    #print(dict_cnt_sorted)
    topk_file = open(os.path.splitext(filename)[0]+'_top%d.csv'%(k),'w')
    for (key,value) in dict_cnt_sorted.items():
        topk_file.write('%s,%s\n' % (key, value))
    topk_file.close()
    



















if __name__ == '__main__':
    #topk(filename = 'ptk_pid_crossday_larger.csv', k=2)
    #topk(filename = 'test.csv', k=2)
    #topk(filename = 'ptk_pid_crossday_larger_200.csv', k=20)
    topk(filename = 'ptk_pid_crossday_larger.csv', k=300)
    
