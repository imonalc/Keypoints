"""
    PROGRAMA PARA CALCULAR KEYPOINTS DE DOS IMAGENES 512x1024, ASI MISMO DE SUS CORRESPONDENCIAS
    POR UN KNN BILATERAL

"""
import sys
import os
import cv2
import open3d as o3d

import sys
import pandas as pd
import numpy as np
import argparse

from random import sample
from scipy.spatial.transform import Rotation as Rot

from utils.coord    import coord_3d
from utils.ransac   import *
from utils.keypoint import *
from utils.metrics  import *
from utils.camera_recovering import *

from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm

sys.path.append(os.getcwd()+'/SPHORB-master')
import build1.sphorb_cpp as sphorb


def main():
    ST = time.time()
    path = "/data/Room/0"
    path_o = os.getcwd()+path + '/O.png' ##
    
    os.chdir('SPHORB-master/')
    print(sphorb.sphorb(path_o, int(12000)))
    os.chdir('../')
            

    GL = time.time()
    print("Time:", GL-ST)
    return

if __name__ == '__main__':
    main()
