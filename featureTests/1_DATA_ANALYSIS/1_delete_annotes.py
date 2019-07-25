import os
def select_intrest_pics(pic_path):
    n_img = len(os.listdir(pic_path))
    print(n_img)
    #assert n_img == 9963, 'VOC2007 should be 9963 samples'
    for i in xrange(n_img):
        if i< 10:
            continue
        del_path = os.path.join(pic_path, '{:06d}.xml'.format(i))
        print('i:%d, num_img:%d, del_path:%s'%(i, n_img, del_path))
        if os.path.exists(del_path):
            os.system('rm %s'%(del_path))

if __name__ == '__main__' :
    select_intrest_pics('VOC2007/Annotations/')
