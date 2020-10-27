import sys
import os
sys.path.insert(0, os.getcwd())
PATH = os.path.dirname(__file__)

import cv2
import matplotlib.pyplot as plt 

from Yolov3.Dataset.dataformat import ReadXML_VOC_Format, readTxt

is_train = True
path = r"D:\Code\Dataset\PASCAL-VOOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012"

train_txt = 'ImageSets/Main/train.txt'
val_txt = 'ImageSets/Main/val.txt'
annotations = "Annotations"
jpegimages = "JPEGImages"

def Init():
    images_path = train_txt if (is_train) else val_txt        
    images_path = readTxt(os.path.join(path, images_path))
    images_path.pop(-1)
    # rawdata format: [path_2_image, path_2_xml]
    rawData = list()
    for each in images_path:
        xml = os.path.join(path, annotations, each + '.xml')
        jpeg = os.path.join(path, jpegimages, each + '.jpg')
        rawData.append([jpeg, xml])
    return rawData

def visualize_all():
    rawData = Init()
    for (path_2_image, path_2_xml) in rawData:
        bboxes, fname = ReadXML_VOC_Format(path_2_xml)
        image = cv2.imread(path_2_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for d in (bboxes):
            name, x1, y1, x2, y2 = d
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2, 1)
            cv2.putText(image, str(name),
                        (x1+10, y1+10 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        plt.imshow(image)
        plt.show()
        
visualize_all()