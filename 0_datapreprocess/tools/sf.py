import os
# list all image files
def list_all_files(dir_name, exts = ['jpg', 'bmp', 'png']):
    result = []
    for dir, sub_dirs, file_names in os.walk(dir_name):
        for file_name in file_names:
            if any(file_name.endswith(ext) for ext in exts):
                result.append(os.path.join(dir, file_name))
    return result


def preprocess(file_b):
    filename = file_b
    pid_ptk_map ={}
    print('Read the file to Dict ... ')
    with open(filename, 'r') as f:
        for line in f.readlines():
            words = line.strip().split(',')
            pid = words[1]
            faceid = words[0]
            if pid not in pid_ptk_map:
                pid_ptk_map[pid] = set([])
            pid_ptk_map[pid].add(faceid)
    print('... done')
    
    print('operate on dict ...')
    nfilename = os.path.splitext(file_b)[0]+'removed.csv'
    nptkfile = open(nfilename, 'w')
    for (key, value) in pid_ptk_map.items():
        #print("key:%s, value:%s"%(key, value))
        # KEY: PID, VALUE: FID
        #nptkfile.write('%s,%s\n'%(value, key[2:]))
        for va in value:
            nptkfile.write('%s,%s\n'%(va, key[2:]))
    nptkfile.close
    print('... done')
    return nfilename

def file_sets(file_a, file_b):
    file_rb = preprocess(file_b)
    s1 = set(open(file_a,'r').readlines())
    s2 = set(open(file_rb,'r').readlines())
    print('ins: %s'%(s1.intersection(s2)))
    print('uni: %s'%(s1.union(s2)))
    print('dif: %s'%(s1.difference(s2).union(s2.difference(s1))))
    
def restore_ptk_baseon_reduced_rawdir(file_a, rawpath):
    #dir_name = 'tstt'
    dir_name = rawpath
    pid_ptk_map ={}
    print('walking dir ...')
    for dir, sub_dirs, file_names in os.walk(dir_name):
        print("dir:%s, sub_dirs:%s, file_names:%s"%(dir, sub_dirs, file_names))
        if dir == dir_name: # 1st loop
            for sdr in sub_dirs:
                pid = sdr[2:]
                if pid not in pid_ptk_map:
                    pid_ptk_map[pid] = set([])
        else:
            pid = os.path.basename(dir)[2:]
            for faceid in file_names:
                pid_ptk_map[pid].add(faceid)
    print('... done')
    print(pid_ptk_map)
    print('operate on dict ...')
    nfilename = os.path.splitext(file_a)[0]+'_reduced.csv'
    nptkfile = open(nfilename, 'w')
    for (key, value) in pid_ptk_map.items():
        #print("key:%s, value:%s"%(key, value))
        # KEY: PID, VALUE: FID
        #nptkfile.write('%s,%s\n'%(value, key[2:]))
        for va in value:
            nptkfile.write('%s,%s\n'%(va, key))
    nptkfile.close
    print('... done')
    return nfilename

def format_topk(file_tpk):
    filename = file_tpk
    nfname = "fmt_"+file_tpk
    nftpk = open(nfname,"w")
    with open(filename, 'r') as f:
        for line in f.readlines():
            words = line.strip().split(',')
            end = words[1].find('_')
            pid = words[1][end+1:]
            faceid = words[0]
            nftpk.write("%s,%s\n"%(faceid, pid))
    nftpk.close()
    return nfname
 
 
def get_file_sets(file_ori, file_topk, file_reduced):
    fmt_file_topk = format_topk(file_topk)
    sori = set(open(file_ori,'r').readlines())
    stopk = set(open(fmt_file_topk,'r').readlines())
    srdc = set(open(file_reduced,'r').readlines())
    '''
    print('ins: %s'%(s1.intersection(s2)))
    print('uni: %s'%(s1.union(s2)))
    print('dif: %s'%(s1.difference(s2).union(s2.difference(s1))))
    '''
    dif_ori_tpk = sori.difference(stopk).union(stopk.difference(sori))
    print("ori - tpk: ", dif_ori_tpk)
    nptk = dif_ori_tpk.union(srdc)
    print("ori - tpk + rdc:", nptk)

def file_i_u_d(f_ori, f_a):
    print('read file to sets ...')
    fori= set(open(f_ori,'r').readlines())
    fa= set(open(f_a,'r').readlines())
    #srdc = set(open(f_b,'r').readlines())
    dif_o_a = fori.difference(fa).union(fa.difference(fori))
    #print(dif_o_a)
    print('... done')
    nfilename = "dif_"+f_ori
    nfile = open(nfilename, 'w')
    for si in dif_o_a:
        nfile.write(si)
    nfile.close()
    
if __name__ == '__main__':
    #freduce = restore_ptk_baseon_reduced_rawdir(file_a="test.csv", rawpath = 'tstt')
    #get_file_sets(file_ori = 'test.csv', file_topk = 'test_top2.csv', file_reduced = freduce)
    #file_sets(file_a="test.csv", file_b=file_b_)#"test_top2.csv" )
    #file_i_u_d(f_ori ="ptk_pid_crossday_larger.csv" , f_a="ptk_pid_crossday_larger_top43.csv" , f_b="")
    fmt_a = format_topk("ptk_pid_crossday_larger_top43.csv")
    print("fmt_a:%s",fmt_a)
    file_i_u_d(f_ori ="ptk_pid_crossday_larger.csv" , f_a=fmt_a)
    #file_i_u_d(f_ori ="test.csv" , f_a="fmt_test_top2.csv" )
