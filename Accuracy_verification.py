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

    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--points', type=int, default = 12000)
    parser.add_argument('--match', default="ratio")
    parser.add_argument('--g_metrics',default="False")
    parser.add_argument('--solver', default="None")
    parser.add_argument('--inliers', default="8PA")
    parser.add_argument('--pose'      , default="pose_test")
    parser.add_argument('--descriptors', nargs='+')
    args = parser.parse_args()

    DESCRIPTORS = args.descriptors
    # ----------------------------------------------
    # Parameters
    # ----------------------------------------------

    NUM = 0
    R_errors, T_errors = [], []
    for i in range(len(DESCRIPTORS)):
        R_errors.append([])
        T_errors.append([])

    if args.g_metrics == "False":
        METRICS = np.zeros((len(DESCRIPTORS),2))
        metrics = ['Matched','Keypoint']
    else:
        METRICS = np.zeros((len(DESCRIPTORS),7))
        metrics = ['Matched','Keypoint','Pmr','Pr','R','Ms','E']

    pose = args.pose
    TIMES = []
    path_true_value = os.path.join(os.getcwd(), "data/Calibration/", pose)
    RQ_true_value = np.genfromtxt(path_true_value + "/RQ_true_value.csv", delimiter=',')
    T_true_value = np.genfromtxt(path_true_value + "/T_true_value.csv", delimiter=',')

    mypath = os.path.join('data/Farm',pose)
    paths  = [os.path.join(os.getcwd(),'data/Farm/',pose,f) for f in listdir('data/Farm/'+pose) if isdir(join(mypath, f))]
    NUM = NUM + len(paths)
    print(paths)

    std = []
    for path in tqdm(paths):

        for indicador, descriptor in enumerate(DESCRIPTORS):


            try:
                opt, mode, sphered = get_descriptor(descriptor)

                base_order = 1  # Base sphere resolution
                sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
                scale_factor = 1.0  # How much to scale input equirectangular image by
                save_ply = False  # Whether to save the PLY visualizations too
                dim = np.array([2*sphered, sphered])

                path_o = path + '/O2.png'
                path_r = path + '/R2.png'


                if opt != 'sphorb':
                    corners = tangent_image_corners(base_order, sample_order)
                    pts1, desc1 = process_image_to_keypoints(path_o, corners, scale_factor, base_order, sample_order, opt, mode)
                    pts2, desc2 = process_image_to_keypoints(path_r, corners, scale_factor, base_order, sample_order, opt, mode)
                    pts1, pts2, desc1, desc2 = sort_key(pts1, pts2, desc1, desc2, args.points)

                else:
                    os.chdir('SPHORB-master/')
                    pts1, desc1 = get_kd(sphorb.sphorb(path_o, args.points))
                    pts2, desc2 = get_kd(sphorb.sphorb(path_r, args.points))
                    os.chdir('../')


                if len(pts1.shape) == 1:
                    pts1 = pts1.reshape(1,-1)
                if len(pts2.shape) == 1:
                    pts2 = pts2.reshape(1,-1)

                if pts1.shape[0] > 0 or pts2.shape[0] >0:
                    s_pts1, s_pts2, x1, x2 = matched_points(pts1, pts2, desc1, desc2, '100p', opt, args.match)
                    
                    x1,x2 = coord_3d(x1, dim), coord_3d(x2, dim)
                    s_pts1, s_pts2 = coord_3d(s_pts1, dim), coord_3d(s_pts2, dim)

                    if x1.shape[0] < 8:
                        R_error, T_error = 3.14, 3.14
                    else:
                        inicio = time.time()
                        if args.solver   == 'None':
                            E, cam = get_cam_pose_by_ransac_8pa(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                        elif args.solver == 'SK':
                            E, can = get_cam_pose_by_ransac_opt_SK(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                        elif args.solver == 'GSM':
                            E, can = get_cam_pose_by_ransac_GSM(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                        elif args.solver == 'GSM_wRT':
                            E, can = get_cam_pose_by_ransac_GSM_const_wRT(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                        elif args.solver == 'GSM_SK':
                            E, can = get_cam_pose_by_ransac_GSM_const_wSK(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                        fin = time.time()
                        TIMES.append(fin-inicio)
                        R1_,R2_,T1_,T2_ = decomposeE(E.T)
                        R_,T_ = choose_rt(R1_,R2_,T1_,T2_,x1,x2)

                        RQ = mat2quat(R_)
                        T_norm = T_ / np.linalg.norm(T_)
                    

                    R_error = 2 * np.arccos(np.dot(RQ, RQ_true_value))
                    T_error = math.acos(np.dot(T_norm, T_true_value))

                    R_errors[indicador].append(R_error)
                    T_errors[indicador].append(T_error)

                    METRICS[indicador,:] = METRICS[indicador,:] + [x1.shape[0], (s_pts1.shape[0]+s_pts2.shape[1])/2]

                    std.append(x1.shape[0])
            except:
                print("Unexpected error")



    print('ALL:')
    print(np.mean(np.array(TIMES)))

    print(R_errors)
    print(T_errors)

    np.savetxt(os.path.join(os.getcwd(),'data/Farm', pose, 'R_errors.csv'), R_errors, delimiter=',')
    np.savetxt(os.path.join(os.getcwd(),'data/Farm', pose, 'T_errors.csv'), T_errors, delimiter=',')

    return




def mat2quat(mat):
    rot = Rotation.from_matrix(mat)
    quat = rot.as_quat()
    return quat


def mean_quat(R_quat_list):
    x = np.array([R_quat for R_quat in R_quat_list])
    m = x.T @ x
    w, v = np.linalg.eig(m)
    return v[:, np.argmax(w)]



def get_descriptor(descriptor):
    if descriptor == 'sphorb':
        return 'sphorb', 'erp', 640
    elif descriptor == 'sift':
        return 'sift', 'erp', 512
    elif descriptor == 'tsift':
        return 'sift', 'tangent', 512
    elif descriptor == 'orb':
        return 'orb', 'erp', 512
    elif descriptor == 'torb':
        return 'orb', 'tangent', 512
    elif descriptor == 'spoint':
        return 'superpoint', 'erp', 512
    elif descriptor == 'tspoint':
        return 'superpoint', 'tangent', 512
    

def sort_key(pts1, pts2, desc1, desc2, points):

    ind1 = np.argsort(pts1[:,2].numpy(),axis = 0)[::-1]
    ind2 = np.argsort(pts2[:,2].numpy(),axis = 0)[::-1]

    max1 = np.min([points,ind1.shape[0]])
    max2 = np.min([points,ind2.shape[0]])

    ind1 = ind1[:max1]
    ind2 = ind2[:max2]

    pts1 = pts1[ind1.copy(),:]
    pts2 = pts2[ind2.copy(),:]

    desc1 = desc1[:,ind1.copy()]
    desc2 = desc2[:,ind2.copy()]

    pts1 = np.concatenate((pts1[:,:2], np.ones((pts1.shape[0],1))), axis = 1 )
    pts2 = np.concatenate((pts2[:,:2], np.ones((pts2.shape[0],1))), axis = 1 )

    desc1 = np.transpose(desc1,[1,0]).numpy()
    desc2 = np.transpose(desc2,[1,0]).numpy()

    return pts1, pts2, desc1, desc2


def get_kd(array):

    array = np.array(array)
    delimiter = int(array[-1])
    A = array[:-1]
    K = A[:delimiter].reshape(-1,3)
    D = A[delimiter:].reshape(-1,32)
    return K,D


def matched_points(pts1, pts2, desc1, desc2, opt, args_opt, match='ratio'):

    if opt[-1] == 'p':
        porce = int(opt[:-1])
        n_key = int(porce/100 * pts1.shape[0])
    else:
        n_key = int(opt)

    s_pts1  = pts1.copy()[:n_key,:]
    s_pts2  = pts2.copy()[:n_key,:]
    s_desc1 = desc1.copy().astype('float32')[:n_key,:]
    s_desc2 = desc2.copy().astype('float32')[:n_key,:]

    if  'orb' in args_opt:
        s_desc1 = s_desc1.astype(np.uint8)
        s_desc2 = s_desc2.astype(np.uint8)

    if match == '2-cross':
        if 'orb' in args_opt:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, True)
        matches = bf.match(s_desc1, s_desc2)
    elif match == 'ratio':
        if 'orb' in args_opt:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, False)
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, False)
        matches = bf.knnMatch(s_desc1,s_desc2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        matches = good

    M = np.zeros((2,len(matches)))
    for ind, match in zip(np.arange(len(matches)),matches):
        M[0,ind] = match.queryIdx
        M[1,ind] = match.trainIdx

    num_M = M.shape[1]

    return s_pts1, s_pts2, s_pts1[M[0,:].astype(int),:3], s_pts2[M[1,:].astype(int),:3]


def AUC(ROT, TRA, MET, L):

    RAUC  = np.zeros(len(L))
    TAUC  = np.zeros(len(L))

    for index, t in enumerate(L):
        ids = np.where(ROT<np.radians(t))[0]
        RAUC[index] = len(ids)/len(ROT)

    for index, t in enumerate(L):
        ids = np.where(TRA<np.radians(t))[0]
        TAUC[index] = len(ids)/len(TRA)

    return RAUC, TAUC, np.array(MET)


if __name__ == '__main__':
    main()