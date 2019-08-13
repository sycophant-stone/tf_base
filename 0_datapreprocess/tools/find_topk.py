import operator
import os 

def topk(filename, k=2):
    #ptk_pid_map = {}
    pid_ptk_map ={}
    print('Read the file ... ')
    with open(filename, 'r') as f:
        for line in f.readlines():
            words = line.strip().split(',')
            pid = words[1]
            faceid = words[0]
            if pid not in pid_ptk_map:
                pid_ptk_map[pid] = set([])
            pid_ptk_map[pid].add(faceid)
    print('... done')

    print('Sorting... ')
    pid_sorted = sorted(pid_ptk_map.items(), key=lambda item:len(item[1]), reverse = True)
    print('... done')

    print('Writing... ')
    topk_file = open(os.path.splitext(filename)[0]+'_top%d.csv'%(k),'w')
    for idx in range(min(k, len(pid_sorted))):
        pid = pid_sorted[idx][0]
        for ptk in pid_ptk_map[pid]:
            topk_file.write('%s,%s_%s\n'%(ptk,idx,pid))
    topk_file.close()
    print('... done')


if __name__ == '__main__':
    #topk(filename = 'ptk_pid_crossday_larger.csv', k=2)
    #topk(filename = 'test.csv', k=2)
    #topk(filename = 'ptk_pid_crossday_larger_200.csv', k=2)
    topk(filename = 'ptk_pid_crossday_larger.csv', k=43)
    
