import sys
import os


File_Path = os.getcwd()
sys.path.append(File_Path)
sys.path.insert(0, os.path.join(os.getcwd(), '../..'))
#sys.path.insert(1, File_Path)

from Yolov3.Utils.train import template_dataLoaderFunc, train
from simple_data import readTxt, simple_data

def loadData(args):
    labels = readTxt(os.path.join(File_Path, 'config', 'class.names'))
    labels.pop(-1)
    return template_dataLoaderFunc(simple_data, args, labels)

if __name__ == "__main__":
    print('Initilizing..')
    train(
        File_Path,
        loadData
    )
    
    
