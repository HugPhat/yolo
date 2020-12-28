# test
from torch.utils.data import DataLoader
#
import sys
import os
sys.path.insert(0, os.getcwd())
PATH = os.path.dirname(__file__)

from Yolov3.Dataset.dataformat import ReadXML_VOC_Format, readTxt
from Yolov3.Dataset.dataset import yoloCoreDataset, cv2, np



class simple_data(yoloCoreDataset):
    def __init__(self, path, labels, 
                    img_size=416,  # fixed size image
                    debug=False,
                    argument=True,
                    draw=False,
                    max_objects=3,
                    is_train=True,
                    split = None
                ):
        super(simple_data, self).__init__(path, labels,
                                       debug=debug, is_train=is_train, split=split)
        self.debug = debug
        self.argument = argument
        self.draw = draw
        self.max_objects = max_objects
        

    def InitDataset(self):
        """
            +data root:
                + annotations
                + images
        """
        annotations = "annotations"
        jpegimages = "images"

        
        images_path = [os.path.split(each)[-1].split('.')[0] 
                        for each in os.listdir(os.path.join(self.path, jpegimages))]
        n_imgs = len(images_path)
        if self.split is None:
            ratio = int(n_imgs*0.8)
        elif self.split < 1. and self.split > 0:
            ratio = int(n_imgs*self.split)
        else:
            raise f'Invalid split ratio of data, must be in range [0,1] but got {self.split}'

        if self.is_train:
            images_path = images_path[:ratio]
        else:
            images_path = images_path[ratio:]

        # rawdata format: [path_2_image, path_2_xml]
        rawData = list()
        for each in images_path:
            xml = os.path.join(self.path, annotations, each + '.xml')
            jpeg = os.path.join(self.path, jpegimages, each + '.png')
            rawData.append([jpeg, xml])
        return rawData
        
    def GetData(self, index, **kwargs):
        path_2_image, path_2_xml = self.rawData[index]
        if self.debug:
            print(f"image path {path_2_image} || xml path {path_2_xml}")
        bboxes, fname = ReadXML_VOC_Format(path_2_xml)
        image = cv2.imread(path_2_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, bboxes, fname

if __name__ == '__main__':
    import torch
    labels = readTxt(os.path.join(PATH, 'config', 'class.names'))
    path_2_root = r"E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\simple_obj_detection\cat_dog"
    #path_2_root = r"D:\Code\Dataset\PASCAL-VOOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012"

    voc = simple_data(path= path_2_root,split=0.8, is_train=True, labels=labels, debug=True, draw=1, argument=True)
    for i, (inp, tar) in enumerate(voc):
        print('INP max {} min {}'.format(torch.max(inp), torch.min(inp)))
        print('TAR max {} min {}'.format(torch.max(tar), torch.min(tar)))
        print(f'{i} / {len(voc)}')
