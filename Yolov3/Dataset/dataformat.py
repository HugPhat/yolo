import xml.etree.cElementTree as ET
import numpy as np

def VOC_Format(path:str):
    """ Read .XML annotation file with VOC Format 

        Return: np.ndarray [[class_name, x1, y1, x2, y2],...]

    Args:
        path (str): [description]
    """
    tree = ET.parse(path)
    root = tree.getroot()
    list_with_all_boxes = []
    for boxes in root.iter('object'):
        filename = root.find('filename').text
        ymin, xmin, ymax, xmax = None, None, None, None
        class_name = boxes.find('name').text
        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)
        list_with_single_boxes = [class_name, xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
    return list_with_all_boxes    

def COCO_Format(path):
    return