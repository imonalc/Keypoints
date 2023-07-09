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
from scipy.spatial.transform import Rotation

sys.path.append(os.getcwd()+'/SPHORB-master')
import build1.sphorb_cpp as sphorb


def main():

    # pose2
    R1=[[-0.985240889462296,-0.05281138968137987,-0.16282305380831452],
    [-0.13196846852588873,-0.37145957767562543,0.9190223639650872],
    [-0.10901703099275527,0.9269459203604962,0.35900772649144597]]
    T1=[-9.549316912279192,-21.266506415074893,23.71880171244876]
    T1=[-16.860834579660512,72.64316665534356,-59.911685139244106]
    R2=[[-0.985240889462296,-0.05281138968137987,-0.16282305380831452],
    [-0.13196846852588873,-0.37145957767562543,0.9190223639650872],
    [-0.10901703099275527,0.9269459203604962,0.35900772649144597]]
    T2=[-9.549316912279192,-21.266506415074893,23.71880171244876]
    R3=[[0.05361816960090926,0.9432839125188384,0.3276286804780685],
    [-0.12766942524069552,-0.31893218157688524,0.9391393833789098],
    [0.9903664017662143,-0.09218310006590624,0.10332795514709714]]
    T3=[-71.34211853451662,-38.393235390652215,-25.0250651425998]
    R4=[[0.05361816960090926,0.9432839125188384,0.3276286804780685],
    [-0.12766942524069552,-0.31893218157688524,0.9391393833789098],
    [0.9903664017662143,-0.09218310006590624,0.10332795514709714]]
    T4=[-71.34211853451662,-38.393235390652215,-25.0250651425998]
    R5=[[0.6745350507537807,0.7244844577989927,-0.14186167809628475],
    [-0.7178901934928802,0.6885245425224815,0.10279895150711642],
    [0.1721514896576542,0.03249961156547833,0.9845342248275291]]
    T5=[-11.169046479742441,-26.735638420390394,7311651.770827603]
    R6=[[0.6745350507537807,0.7244844577989927,-0.14186167809628475],
    [-0.7178901934928802,0.6885245425224815,0.10279895150711642],
    [0.1721514896576542,0.03249961156547833,0.9845342248275291]]
    T6=[-11.169046479742441,-26.735638420390394,7311651.770827603]
    R7=[[0.9736790581932009,0.11348077125791174,-0.19766437764734132],
    [-0.1149624102217902,0.9933617828307056,0.004001580612571423],
    [0.1968063410360502,0.018827718027195794,0.9802615881293576]]
    T7=[38.03430341250563,1.6819605454951179,-9.188640292581667]
    R8=[[0.9736790581932009,0.11348077125791174,-0.19766437764734132],
    [-0.1149624102217902,0.9933617828307056,0.004001580612571423],
    [0.1968063410360502,0.018827718027195794,0.9802615881293576]]
    T8=[38.03430341250563,1.6819605454951179,-9.188640292581667]
    R9=[[0.973058710796878,0.03964063588315327,-0.22712411877450808],
    [-0.046784974395938185,0.9985624565839489,-0.026156958382052595],
    [0.22576073952990364,0.03607825228317309,0.9735144827885729]]
    T9=[47.64692477152729,-4.877416545321482,20.077061277852234]
    R10=[[0.973058710796878,0.03964063588315327,-0.22712411877450808],
    [-0.046784974395938185,0.9985624565839489,-0.026156958382052595],
    [0.22576073952990364,0.03607825228317309,0.9735144827885729]]
    T10=[47.64692477152729,-4.877416545321482,20.077061277852234]

    R1_quat = mat2quat(R1)
    R2_quat = mat2quat(R2)
    R3_quat = mat2quat(R3)
    R4_quat = mat2quat(R4)
    R5_quat = mat2quat(R5)
    R6_quat = mat2quat(R6)
    R7_quat = mat2quat(R7)
    R8_quat = mat2quat(R8)
    R9_quat = mat2quat(R9)
    R10_quat = mat2quat(R10)

    R_quat_l = [R1_quat, R2_quat, R3_quat, R4_quat, R5_quat, R6_quat, R7_quat, R8_quat, R9_quat, R10_quat]
    T_l = [T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]


    RQ_true_value = mean_quat(R_quat_l)
    T_true_value = get_T_true_value(T_l)

    print(RQ_true_value)
    print(T_true_value)

    np.savetxt(os.path.join(os.getcwd(),'RQ_true_value.csv'), RQ_true_value, delimiter=',')
    np.savetxt(os.path.join(os.getcwd(),'T_true_value.csv'), T_true_value, delimiter=',')

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