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
import utils.calibrate_SPHcamera_opencv as calibrate_SPHcamera_opencv

from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm
from scipy.spatial.transform import Rotation

sys.path.append(os.getcwd()+'/SPHORB-master')
import build1.sphorb_cpp as sphorb


def main():

    # pose1
    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--pose'      , default="pose_test")
    parser.add_argument('--descriptors', nargs='+')
    args = parser.parse_args()

    DESCRIPTORS = args.descriptors
    # ----------------------------------------------
    # Parameters
    # ----------------------------------------------

    pose = args.pose
    mypath = os.path.join('data/Calibration',pose)
    paths  = [os.path.join(os.getcwd(),'data/Calibration',pose,f) for f in listdir('data/Calibration/'+pose) if isdir(join(mypath, f))]
    NUM = 0
    NUM = NUM + len(paths)
    print(paths)

    R_quat_l = []
    T_l = []

    for path in tqdm(paths):
        R, T = calibrate_SPHcamera_opencv.calib_sph(path)
        if abs(R[0][0]) < 1e-3 or np.isnan(R).any():
            continue
        R = R.tolist()
        T = T.tolist()
        T = [T[0][0], T[1][0], T[2][0]]
        R_quat_l.append(mat2quat(R))
        T_l.append(T)


    RQ_true_value = mean_quat(R_quat_l)
    T_true_value = get_T_true_value(T_l)

    print(RQ_true_value)
    print(T_true_value)

    np.savetxt(os.path.join(os.getcwd(),'RQ_true_value.csv'), RQ_true_value, delimiter=',')
    np.savetxt(os.path.join(os.getcwd(),'T_true_value.csv'), T_true_value, delimiter=',')
    print("move files")

    return

def get_T_true_value(T_list):
    T_filtered_list = []
    for i in range(len(T_list)):
        if abs(T_list[i][0]) >100: continue
        if abs(T_list[i][1]) >100: continue
        if abs(T_list[i][2]) >100: continue
        T_filtered_list.append(T_list[i]/np.linalg.norm(T_list[i]))
        T_norm_array = np.array([T for T in T_filtered_list])
        return np.mean(T_norm_array, axis=0) 

def mat2quat(mat):
    rot = Rotation.from_matrix(mat)
    quat = rot.as_quat()
    return quat


def mean_quat(R_quat_list):
    x = np.array([R_quat for R_quat in R_quat_list])
    m = x.T @ x
    w, v = np.linalg.eig(m)
    return v[:, np.argmax(w)]

if __name__ == '__main__':
    main()