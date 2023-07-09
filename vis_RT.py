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

sys.path.append(os.getcwd()+'/fivepoint')
sys.path.append(os.getcwd()+'/SPHORB-master')
import build.fivep as f
import build1.sphorb_cpp as sphorb


def main():

    R_ = [
        [-0.59239975,  0.57356478,  0.56576141],
        [ 0.5599638,   0.79802506, -0.2227028 ],
        [-0.57922627,  0.18487683, -0.79392537],
    ]
    T_ = [
        -7.18727636e+01, 3.20189233e+00, 1.06771598e+06
    ]
    vis_posture(R_, T_)
    return 




def vis_posture(R, T):
    axis_a = o3d.geometry.TriangleMesh.create_coordinate_frame()
    axis_b = o3d.geometry.TriangleMesh.create_coordinate_frame()
    T_a = np.eye(4) # Camera pose A
    T_b = np.pad(R, ((0, 1), (0, 1)), mode='constant') #Camera pose B
    for idx in range(3): T_b[idx][3] = T[idx]
    T_b[3][3] = 1
    print(T_b)

    axis_a.transform(T_a)
    axis_b.transform(T_b)
    o3d.visualization.draw_geometries([axis_a, axis_b])

if __name__ == '__main__':
    main()
