'''
Come From https://github.com/yangyi02/densebox
densebox\experiment\kitti\prepare_data\python\prepare.py
'''

def get_kitti_train(label_dir):
    """
    Given the directory of raw kitti labels, output an array of dictionaries,
    where each dictionary contain the information of an image.

    Input
        label_dir   -  directory for raw kitti labels

    Output
        annos     -  a list of dictionary
            type        -  element, i.e. 'Pedestrian'
            truncation  -  element, i.e. 0
            occlusion   -  element, i.e. 0
            alpha       -  element, i.e. -0.2000
            x1          -  element, i.e. 712.4000
            y1          -  element, i.e. 143
            x2          -  element, i.e. 810.7300
            y2          -  element, i.e. 307.9200
            h           -  element, i.e. 1.8900
            w           -  element, i.e. 0.4800
            l           -  element, i.e. 1.2000
            t           -  element, i.e. [1.8400 1.4700 8.4100]
            ry          -  element, i.e. 0.0100
    """
    # get total number of images
    n_img = len(os.listdir(label_dir))
    assert n_img == 7481, 'Kitti car detection dataset should contain totally 7481 training images'

    annos = []
    for i in xrange(n_img):
        label_path = os.path.join(label_dir, '{:06d}.txt'.format(i))
        with open(label_path) as fio:
            lines = fio.read().splitlines()

        num_line = len(lines)
        anno_i = []
        for line_id in xrange(num_line):
            line = lines[line_id]
            terms = line.split()
            assert len(terms) == 15

            anno = EasyDict()
            anno.type = terms[0]
            anno.truncation = float(terms[1])
            anno.occlusion = int(terms[2])
            anno.alpha = float(terms[3])
            anno.x1 = float(terms[4])
            anno.y1 = float(terms[5])
            anno.x2 = float(terms[6])
            anno.y2 = float(terms[7])
            anno.h = float(terms[8])
            anno.w = float(terms[9])
            anno.l = float(terms[10])
            anno.t = [float(terms[11 + j]) for j in xrange(3)]
            anno.ry = float(terms[14])
            anno_i.append(anno)

        annos.append(anno_i)

    return anno

