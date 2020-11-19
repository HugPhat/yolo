import json
import numpy as np
import xml.etree.cElementTree as ET

class COCO:
    def __init__(self, path_to_annot, path_to_images=None) -> None:
        self.path_to_images = path_to_images
        self.Init(path_to_annot)

    def Init(self, path_to_annot):
        self.images = {}
        self.classes_names = {}
        annots = {}
        # load json
        with open(path_to_annot, 'r') as f:
          data = json.load(f)
        #process class names
        for each in data["categories"]:
          id = each["id"]
          each.pop("id")
          self.classes_names.update({id: each})
        # get image info by id
        for each in data["images"]:
          image_id = each["id"]
          each.pop("id")
          self.images.update({image_id: each})
        # add annot to images
        for each in data["annotations"]:
          image_id = each["image_id"]
          cat_id = each["category_id"]
          each.update({"class": self.classes_names[cat_id]})
          each.pop("image_id")
          self.images[image_id].update(each)

    def __iter__(self):
      for k, v in self.images.items():
          yield (k, v)

    def __getitem__(self, index) -> dict:
      try:
        data = self.images[index]
      except KeyError:
        print("Key Error : {}".format(index))
        data = None
      return data

def readTxt(path):
    with open(path, 'r') as f:
        data = f.read()
    return data.split('\n')

def ReadXML_VOC_Format(path:str):
    """ Read .XML annotation file with VOC Format 

        Return: list [[class_name, x1, y1, x2, y2],...]

    Args:
        path (str): path to xml file
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
    return list_with_all_boxes, filename  

def COCO_Format(path_to_annot, path_to_images):
    """ Read Annotation

    Args:
        path_to_annot ([str]): [path to annotation]
        path_to_images ([str]): [path to images folder]
    """
    with open(path_to_annot, 'r') as f:
        data = 
