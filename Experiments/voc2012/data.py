from Yolov3.Dataset.dataformat import ReadXML_VOC_Format, readTxt
import sys
import os
sys.path.insert(0, os.getcwd())

from Yolov3.Dataset.dataset import * 

PATH = os.path.dirname(__file__)

class VOC_data(yoloCoreDataset):
    def __init__(self, path, labels, is_train=True):
        super(VOC_data, self).__init__(path, labels)
        

    def InitDataset(self):
        """Use preset data set in ./ImageSets/Main contains [train.txt, val.txt]
        """
        train_txt = 'ImageSets/Main/train.txt'
        val_txt = 'ImageSets/Main/val.txt'
        annotations = "Annotations"
        jpegimages = "JPEGImages"
        images_path = train_txt if (self.is_train) else val_txt        
        images_path = readTxt(os.path.join(self.path, images_path))
        # rawdata format: [path_2_image, path_2_xml]
        rawData = list()
        for each in images_path:
            xml = os.path.join(self.path, annotations, each + '.xml')
            jpeg = os.path.join(self.path, jpegimages, each + '.jpg')
            rawData.append([jpeg, xml])
        return rawData
        
    def GetData(self, index, **kwargs):
        path_2_image, path_2_xml = self.rawData[index]
        bboxes = ReadXML_VOC_Format(path_2_xml)
        image = cv2.imread(path_2_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, bboxes
        
labels = readTxt(os.path.join(PATH, 'config', 'class.names'))
path_2_root = r"E:\ProgrammingSkills\python\DEEP_LEARNING\DATASETS\PASCALVOC\VOCdevkit\VOC2012"

voc = VOC_data(path= path_2_root, labels=labels)
