from lxml import etree
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import os
import cv2

class PascalVocAnn(object):
    def __init__(self, image_name="", xml=""):
        if len(xml) != 0:
            self.node_root = etree.parse(xml).getroot()
            self.node_filename = self.node_root.find('filename')
            self.node_size = self.node_root.find("size")
            self.node_width = self.node_size.find("width")
            self.node_height = self.node_size.find("height")
            self.node_depth = self.node_size.find("depth")
            
        else:
            image_name = os.path.abspath(image_name)
            self.node_root = Element('annotation')
            self.node_filename = SubElement(self.node_root, 'filename')
            self.node_filename.text = image_name
            h,w,c = cv2.imread(image_name).shape
            self.node_size = SubElement(self.node_root, "size")
            self.node_width = SubElement(self.node_size, "width")
            self.node_width.text = str(w)
            self.node_height = SubElement(self.node_size, "height")
            self.node_height.text = str(h)
            self.node_depth = SubElement(self.node_size, "depth")
            self.node_depth.text = str(c)

    def set_filename(self, file_name=""):
        self.node_filename.text = file_name

    def set_size(self, size=[]):
        self.node_height.text = str(size[0])
        self.node_width.text = str(size[1])
        self.node_depth.text = str(size[2])

    def get_size(self):
        return (int(self.node_height.text), int(self.node_width.text), int(self.node_depth.text))
    
    def add_object(self, object_class="head", xmin=0, ymin=0, xmax=0, ymax=0):
        object_new = SubElement(self.node_root, 'object')
        object_name_new = SubElement(object_new, 'name')
        object_name_new.text = object_class
        object_difficult = SubElement(object_new, 'difficult')
        object_difficult.text = '0'

        node_bndbox = SubElement(object_new, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(xmin)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(ymin)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(xmax)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(ymax)

    def write_xml(self, xml_path=""):
        xml_str = tostring(self.node_root, pretty_print=True)
        dom = parseString(xml_str)
        with open(xml_path, 'wb') as f:
            f.write(dom.toprettyxml(encoding="utf-8"))

    def get_boxes(self):
        boxes=[]
        objs = self.node_root.findall('object')
        for obj in objs:
            label = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            boxes.append([label, xmin, ymin, xmax, ymax])
        return boxes

    def check_boxes(self):
        objs = self.node_root.findall('object')
        for obj in objs:
            label = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            h,w,c = self.get_size()
            xmin = xmin if xmin >= 0 else 0
            ymin = ymin if ymin >= 0 else 0
            xmax = xmax if xmax < w else w - 1
            ymax = ymax if ymax < h else h - 1
            bndbox.find("xmin").text = str(xmin)
            bndbox.find("ymin").text = str(ymin)
            bndbox.find("xmax").text = str(xmax)
            bndbox.find("ymax").text = str(ymax)
        

if __name__ == "__main__":
    pascal_voc_ann = PascalVocAnn(xml="./Benchmark/Annotations/ch00005_20190206092957.mp4.cut.mp4_001500.xml")
    print(pascal_voc_ann.get_boxes())
