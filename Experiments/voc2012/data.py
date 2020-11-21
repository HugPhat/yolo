# test
from torch.utils.data import DataLoader
#
import sys
import os
sys.path.insert(0, os.getcwd())
PATH = os.path.dirname(__file__)

from Yolov3.Dataset.dataformat import ReadXML_VOC_Format, readTxt
from Yolov3.Dataset.dataset import yoloCoreDataset, cv2, np



class VOC_data(yoloCoreDataset):
    def __init__(self, path, labels, 
                    img_size=416,  # fixed size image
                    debug=False,
                    argument=True,
                    draw=False,
                    max_objects=15,
                    is_train=True
                ):
        super(VOC_data, self).__init__(path, labels, debug=debug, is_train=is_train)
        self.debug = debug
        self.argument = argument
        self.draw = draw
        self.max_objects = max_objects
        

    def InitDataset(self):
        """Use preset data set in './ImageSets/Main' contains [train.txt, val.txt]
            +data root:
                + /ImageSets/Main:
                                + train.txt
                                + val.txt
                + Annotations
                + JPEGImages
        """
        train_txt = 'ImageSets/Main/train.txt'
        val_txt = 'ImageSets/Main/val.txt'
        annotations = "Annotations"
        jpegimages = "JPEGImages"
        images_path = train_txt if (self.is_train) else val_txt        
        images_path = readTxt(os.path.join(self.path, images_path))
        images_path.pop(-1)
        # rawdata format: [path_2_image, path_2_xml]
        rawData = list()
        for each in images_path:
            xml = os.path.join(self.path, annotations, each + '.xml')
            jpeg = os.path.join(self.path, jpegimages, each + '.jpg')
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

    labels = readTxt(os.path.join(PATH, 'config', 'class.names'))
    labels.insert(0,0)
    path_2_root = r"E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\PASCALVOC\VOCdevkit\VOC2012"
    #path_2_root = r"D:\Code\Dataset\PASCAL-VOOC\VOCtrainval_11-May-2012\VOCdevkit\VOC2012"

    voc = VOC_data(path= path_2_root, labels=labels, debug=True, draw=False, argument=True)
    for i, each in enumerate(voc):
        print(f'{i} / {len(voc)}')
