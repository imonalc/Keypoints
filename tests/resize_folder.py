import sys
import os
sys.path.append(os.getcwd()+'/fivepoint')
import build.fivep as f

import time
import torch
import torch.nn.functional as F
from spherical_distortion.functional import create_tangent_images, unresample
from spherical_distortion.util import *
import matplotlib.pyplot as plt
from skimage import io
import os
import cv2

import sys
import pandas as pd
import numpy as np
import _spherical_distortion_ext._mesh as _mesh
import argparse

from random import sample
import imageio
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

    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--points', type=int, default = 12000)
    parser.add_argument('--data'      , default="Calibration/pose_test")
    args = parser.parse_args()


    data = args.data
    NUM = 0
    mypath = os.path.join('data',data)
    paths  = [os.path.join(os.getcwd(),'data',data,f) for f in listdir('data/'+data) if isdir(join(mypath, f))]
    NUM = NUM + len(paths)
    print(NUM)

    for path in tqdm(paths):
        path_o2 = path + '/O2.png'
        path_r2 = path + '/R2.png'

        if not os.path.isfile(path_o2):
            path_o = path + '/O.png'
            img_o = cv2.imread(path_o)
            img_o2 = cv2.resize(img_o, (1024, 512))
            cv2.imwrite(path_o2, img_o2)
                
        if not os.path.isfile(path_r2):
            path_r = path + '/R.png'
            img_r = cv2.imread(path_r)
            img_r2 = cv2.resize(img_r, (1024, 512))
            cv2.imwrite(path_r2, img_r2)
    
    return



if __name__ == '__main__':
    main()