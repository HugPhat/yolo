import sys
import os
sys.path.insert(0, os.getcwd())
PATH = os.path.dirname(__file__)

import cv2
import matplotlib.pyplot as plt 

from Yolov3.Dataset.dataformat import ReadXML_VOC_Format, readTxt

is_train = True
path = r"E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\simple_obj_detection\cat_dog"

annotations = "annotations"
jpegimages = "images"

def Init():       
    images_path = os.listdir(os.path.join(path, jpegimages))
    
    # rawdata format: [path_2_image, path_2_xml]
    rawData = list()
    for each in images_path:
        each = os.path.split(each)[-1].split('.')[0]
        xml = os.path.join(path, annotations, each + '.xml')
        jpeg = os.path.join(path, jpegimages, each + '.png')
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
        


def amount_obj_per_class():
    import json 

    rawData = Init()
    labels = readTxt(r'Experiments\simple\config\class.names')
    labels.pop(-1)
    object_per_class = {'objects' : None}
    object_per_class['objects'] = {k: 0 for k in labels}
    object_per_class.update({'sum' : 0})
    bbox_sizes = {'w' : [], 'h' : []}
    for (path_2_image, path_2_xml) in rawData:
        bboxes, fname = ReadXML_VOC_Format(path_2_xml)
        for d in (bboxes):
            name, x1, y1, x2, y2 = d
            #
            object_per_class['objects'][name] += 1
            object_per_class['sum'] += 1
            #
            bbox_sizes['w'].append( x2 - x1 )
            bbox_sizes['h'].append( y2 - y1 )
            #
    
    path = r'Experiments\simple\config'
    with open(os.path.join(path, 'objects.json'), 'w') as f :
        json.dump(object_per_class, f, sort_keys=True, indent=4)
    with open(os.path.join(path, 'bboxes.json'), 'w') as f:
        json.dump(bbox_sizes, f, sort_keys=True, indent=4)

    
      

if __name__ == '__main__':
    #visualize_all()
    amount_obj_per_class()
