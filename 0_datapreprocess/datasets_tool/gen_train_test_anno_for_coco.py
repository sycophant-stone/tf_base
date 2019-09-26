"""gen train, test set's anno for coco
by spliting origin Anno to train's and test's
"""

import os
import shell

def gen_split_train_and_test(main_anno_path, set_txt):
    '''gen split train and test
    '''
    set_dirname = 'Annotations_%s'%(os.path.splitext(set_txt)[0])
    if not os.path.exists(set_dirname):
        os.mkdir(set_dirname)

    with open(set_txt, 'r') as f:
        for line in f.readlines():
            words = line.strip().split(' ')[1].split('/')[1:]
            set_anno_path = '/'.join(words)
            cmd = "cp -rf %s %s/"%(set_anno_path, set_dirname)            
            shell.run_system_command(cmd)


if __name__ == '__main__':
    gen_split_train_and_test(main_anno_path='Annotations', set_txt='test.txt')
    gen_split_train_and_test(main_anno_path='Annotations', set_txt='trainval.txt')
