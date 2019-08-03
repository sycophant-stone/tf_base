import cv2
import os
#ch01014_20190308_ch01014_20190308084110.mp4.cut.mp4_009000.jpg
from pascal_voc import PascalVocAnn

REQ_IMGSIZE=300#128
IM_RUS=REQ_IMGSIZE/2
def crop_and_resize(src_filename, dst_filename):
    #img_filepath = 'ch01014_20190308_ch01014_20190308084110.mp4.cut.mp4_009000.jpg'
    #lb_filepath = 'ch01014_20190308_ch01014_20190308084110.mp4.cut.mp4_009000.xml'
    img_filepath = os.path.splitext(src_filename)[0]+'.jpg'
    lb_filepath  = os.path.splitext(src_filename)[0]+'.xml'
    img = cv2.imread(img_filepath)
    pascal_voc_ann = PascalVocAnn(xml=lb_filepath)
    bboxes = pascal_voc_ann.get_boxes()
    vec = []
    for i,b in enumerate(bboxes):
        xmin, ymin, xmax, ymax = b[1:5]
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        sz = int(max(w, h) * 0.62)
        x = int(xmin + (w - sz) * 0.5)
        y = int(ymin + h - sz)
        vec.extend([x, y, sz, sz])
        xc = x+sz/2
        yc = y+sz/2
        img_xmn = xc-IM_RUS if xc-IM_RUS>=0 else 0
        img_ymn = yc-IM_RUS if yc-IM_RUS>=0 else 0
        img_xmx = xc+IM_RUS if xc+IM_RUS<img.shape[1] else img.shape[1] - 1
        img_ymx = yc+IM_RUS if yc-IM_RUS<img.shape[0] else img.shape[0] - 1
        ioregion = img[img_ymn:img_ymx,img_xmn:img_xmx]
        #print(img_filepath.split('.'))
        #print(os.path.splitext(img_filepath))
        crop_img_name = os.path.splitext(img_filepath)[0]+"_crop_%d.jpg"%(i)
        cv2.imwrite(crop_img_name, ioregion)
        
        nxmin = x- img_xmn
        nymin = y- img_ymn
        nxmax = x+sz-1 - img_xmn
        nymax = y+sz-1 - img_ymn 
        crop_xml_name = os.path.splitext(crop_img_name)[0]+".xml" 
        newpascal_ann = PascalVocAnn(image_name=crop_img_name)
        newpascal_ann.set_filename(file_name=crop_img_name)
        newpascal_ann.set_size(size=[REQ_IMGSIZE, REQ_IMGSIZE, img.shape[2]])
        newpascal_ann.add_object(object_class="head", xmin=nxmin, ymin=nymin, xmax=nxmax, ymax=nymax)
        newpascal_ann.write_xml(crop_xml_name)
        















if __name__ == '__main__':
    crop_and_resize(src_filename, dst_filename)
