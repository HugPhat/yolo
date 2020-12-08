import sys
import os


File_Path = os.getcwd()
sys.path.append(File_Path)
sys.path.insert(0, os.path.join(os.getcwd(), '../..'))
#sys.path.insert(1, File_Path)

from Yolov3.Utils.train import template_dataLoaderFunc, train
import voc_data

def loadData(args):
    labels = voc_data.readTxt(os.path.join(File_Path, 'config', 'class.names'))
    return template_dataLoaderFunc(voc_data.VOC_data, args, labels)

if __name__ == "__main__":
    print('Initilizing..')
    train(
        File_Path,
        loadData
    )
    
    
