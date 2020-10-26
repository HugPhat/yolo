import sys
import os
sys.path.insert(0, os.getcwd())

from Yolov3.Dataset.dataformat import VOC_Format, readTxt
from Yolov3.Dataset.dataset import * 

PATH = os.path.dirname(__file__)

class VOC_data(yoloCoreDataset):
    def __init__(self, path, labels, is_train=True):
        super(VOC_data, self).__init__(path, labels)
        self.is_train = is_train

    def InitDataset(self):
        """Use preset data set in ./ImageSets/Main contains [train.txt, val.txt]
        """
        train_txt = 'ImageSets/Main/train.txt'
        val_txt = 'ImageSets/Main/val.txt'
        annotations = "Annotations"
        jpegimages = "JPEGImages"
        images_path = train_txt if (self.is_train) else val_txt
        
        images_path = readTxt(os.path.join(self.path, images_path))
        
        

voc = VOC_data('','')