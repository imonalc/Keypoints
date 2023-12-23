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
    parser.add_argument('--points', type=int, default = 500)
    parser.add_argument('--match', default="ratio")
    parser.add_argument('--g_metrics',default="False")
    parser.add_argument('--solver', default="GSM_wRT")
    parser.add_argument('--inliers', default="8PA")
    parser.add_argument('--pose'      , default="pose_test")
    parser.add_argument('--descriptors', nargs='+')
    args = parser.parse_args()

    DESCRIPTORS = args.descriptors
    # ----------------------------------------------
    # Parameters
    # ----------------------------------------------

    NUM = 0
    R_ERROR, T_ERROR, TIMES_FP, TIMES_MC, TIMES_PE = [], [], [], [], []
    for i in range(len(DESCRIPTORS)):
        R_ERROR.append([])
        T_ERROR.append([])
        TIMES_FP.append([])
        TIMES_MC.append([])
        TIMES_PE.append([])


    METRICS = np.zeros((len(DESCRIPTORS),2))

    pose = args.pose
    TIMES = []
    st = time.time()
    path_true_value = os.path.join("./data/Farm/new/Calibration/", pose)
    RQ_true_value = np.genfromtxt(path_true_value + "/RQ_true_value.csv", delimiter=',')
    T_true_value = np.genfromtxt(path_true_value + "/T_true_value.csv", delimiter=',')
    Rx = np.genfromtxt(path_true_value + "/Rm_true_value.csv", delimiter=',')
    Tx = np.genfromtxt(path_true_value + "/T_true_value.csv", delimiter=',')

    mypath = os.path.join('./data/Farm/new',pose)
    paths  = [os.path.join('./data/Farm/new/',pose,f) for f in listdir('./data/Farm/new/'+pose) if isdir(join(mypath, f))]
    print(os.path.join('./data/Farm/new',pose))
    NUM = NUM + len(paths)
    print(paths)

    std = []
    for path in tqdm(paths):
        #print(path)

        for indicador, descriptor in enumerate(DESCRIPTORS):


            try:
                opt, mode, sphered, use_our_method = get_descriptor(descriptor)
                base_order = 0  # Base sphere resolution
                sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
                scale_factor = 1.0  # How much to scale input equirectangular image by
                save_ply = False  # Whether to save the PLY visualizations too
                dim = np.array([2*sphered, sphered])
                path_o = path + '/O.png'
                path_r = path + '/R.png'
                if opt != 'sphorb':
                    # ----------------------------------------------
                    # Compute necessary data
                    # ----------------------------------------------
                    # 80 baricenter points
                    corners = tangent_image_corners(base_order, sample_order)
                    t_featurepoint_b = time.perf_counter()
                    pts1, desc1 = process_image_to_keypoints(path_o, corners, scale_factor, base_order, sample_order, opt, mode)
                    pts2, desc2 = process_image_to_keypoints(path_r, corners, scale_factor, base_order, sample_order, opt, mode)
                    t_featurepoint_a = time.perf_counter()
                    pts1, pts2, desc1, desc2 = sort_key(pts1, pts2, desc1, desc2, args.points)
                    
                else:
                    os.chdir('SPHORB-master/')
                    t_featurepoint_b = time.perf_counter()
                    pts1, desc1 = get_kd(sphorb.sphorb(path_o, args.points))
                    pts2, desc2 = get_kd(sphorb.sphorb(path_r, args.points))
                    t_featurepoint_a = time.perf_counter()
                    os.chdir('../')
                
                print(pts1.shape, desc1.shape)

                height_threshold = 512*0.9
                cond1_1 = (pts1[:, 1] < height_threshold)
                cond2_1 = (pts2[:, 1] < height_threshold)
                if pose == "pose1":
                    cond1_2 = ~(((400 < pts1[:, 0]) &(pts1[:, 0] < 840))  & (pts1[:, 1] > 400))
                    cond1_3 = ~(((680 < pts1[:, 0]) &(pts1[:, 0] < 800)) & ((220< pts1[:, 1])))
                    cond2_2 = ~((pts2[:, 0] < 400) & (pts2[:, 1] > 350))
                    cond2_3 = ~(((220 < pts2[:, 0]) &(pts2[:, 0] < 320)) & ((250< pts2[:, 1])))
                elif pose == "pose2":
                    cond1_2 = ~(((500 < pts1[:, 0]) &(pts1[:, 0] < 1000))  & (pts1[:, 1] > 360))
                    cond1_3 = ~(((800 < pts1[:, 0]) &(pts1[:, 0] < 1000)) & ((220< pts1[:, 1])))
                    cond2_2 = ~((pts2[:, 0] < 400) & (pts2[:, 1] > 350))
                    cond2_3 = ~(((100 < pts2[:, 0]) &(pts2[:, 0] < 250)) & ((250< pts2[:, 1])))
                elif pose == "pose3":
                    cond1_2 = ~(((200 < pts1[:, 0]) &(pts1[:, 0] < 700))  & (pts1[:, 1] > 360))
                    cond1_3 = ~(((500 < pts1[:, 0]) &(pts1[:, 0] < 750)) & ((240< pts1[:, 1])))
                    cond2_2 = ~((pts2[:, 0] < 400) & (pts2[:, 1] > 300))
                    cond2_3 = ~(((200 < pts2[:, 0]) &(pts2[:, 0] < 400)) & ((250< pts2[:, 1])))
                elif pose == "pose4":
                    cond1_2 = ~(((500 < pts1[:, 0]) &(pts1[:, 0] < 1000))  & (pts1[:, 1] > 360))
                    cond1_3 = ~(((800 < pts1[:, 0]) &(pts1[:, 0] < 1000)) & ((240< pts1[:, 1])))
                    cond2_2 = ~((pts2[:, 0] < 300) & (pts2[:, 1] > 300))
                    cond2_3 = ~(((60 < pts2[:, 0]) &(pts2[:, 0] < 240)) & ((250< pts2[:, 1])))
                elif pose == "pose5":
                    cond1_2 = ~(((400 < pts1[:, 0]) &(pts1[:, 0] < 900))  & (pts1[:, 1] > 360))
                    cond1_3 = ~(((650 < pts1[:, 0]) &(pts1[:, 0] < 850)) & ((240< pts1[:, 1])))
                    cond2_2 = ~((pts2[:, 0] < 400) & (pts2[:, 1] > 400))
                    cond2_3 = ~(((100 < pts2[:, 0]) &(pts2[:, 0] < 350)) & ((250< pts2[:, 1])))
    
                valid_idx1 = cond1_1 & cond1_2 &cond1_3
                pts1 =  pts1[valid_idx1]
                desc1 = desc1[valid_idx1]
                valid_idx2 = cond2_1 & cond2_2 &cond2_3
                pts2 =  pts2[valid_idx2]
                desc2 = desc2[valid_idx2]


                if len(pts1.shape) == 1:
                    pts1 = pts1.reshape(1,-1)
                if len(pts2.shape) == 1:
                    pts2 = pts2.reshape(1,-1)


                if pts1.shape[0] > 0 or pts2.shape[0] >0:
                    t_matching_b = time.perf_counter()
                    s_pts1, s_pts2, x1, x2 = matched_points(pts1, pts2, desc1, desc2, "100p", opt, args.match, use_new_method=use_our_method)
                    t_matching_a = time.perf_counter()
                    x1,x2 = coord_3d(x1, dim), coord_3d(x2, dim)
                    s_pts1, s_pts2 = coord_3d(s_pts1, dim), coord_3d(s_pts2, dim)
                    
                    if x1.shape[0] < 8:
                        R_error, T_error = 3.14, 3.14
                    else:
                        t_poseestimate_b = time.perf_counter()
                        if args.solver   == 'None':
                            E, cam = get_cam_pose_by_ransac(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                        elif args.solver == 'SK':
                            E, can = get_cam_pose_by_ransac_opt_SK(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                        elif args.solver == 'GSM':
                            E, can = get_cam_pose_by_ransac_GSM(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                        elif args.solver == 'GSM_wRT':
                            E, can, inlier_idx = get_cam_pose_by_ransac_GSM_const_wRT(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                        elif args.solver == 'GSM_SK':
                            E, can, inlier_idx = get_cam_pose_by_ransac_GSM_const_wSK(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                        t_poseestimate_a = time.perf_counter()
                        #print("True:", sum(inlier_idx), len(inlier_idx), ", ratio:", sum(inlier_idx) / len(inlier_idx))
                        R1_,R2_,T1_,T2_ = decomposeE(E.T)
                        R_,T_ = choose_rt(R1_,R2_,T1_,T2_,x1,x2)
                        

                        #print(R_)
                        #print(T_)

                    
                    R_error, T_error = r_error(Rx,R_), t_error(Tx,T_)

                    R_ERROR[indicador].append(R_error)
                    T_ERROR[indicador].append(T_error)
                    TIMES_FP[indicador].append((t_featurepoint_a-t_featurepoint_b)/2)
                    TIMES_MC[indicador].append(t_matching_a-t_matching_b)
                    TIMES_PE[indicador].append(t_poseestimate_a-t_poseestimate_b)
                    METRICS[indicador,:] = METRICS[indicador,:] + [x1.shape[0], (s_pts1.shape[0]+s_pts2.shape[1])/2]
                    std.append(x1.shape[0])
            except:     
                print("Unexpected error:",indicador, opt, use_our_method)
    for indicador, descriptor in enumerate(DESCRIPTORS):
        os.system('mkdir -p '+f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver)
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/R_ERRORS.csv',np.array(R_ERROR[indicador]),delimiter=",")
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/T_ERRORS.csv',np.array(T_ERROR[indicador]),delimiter=",")
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/TIMES_FP.csv',np.array(TIMES_FP[indicador]),delimiter=",")
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/TIMES_MC.csv',np.array(TIMES_MC[indicador]),delimiter=",")
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/TIMES_PE.csv',np.array(TIMES_PE[indicador]),delimiter=",")
print('finish')




def mat2quat(mat):
    rot = Rotation.from_matrix(mat)
    quat = rot.as_quat()
    return quat


def mean_quat(R_quat_list):
    x = np.array([R_quat for R_quat in R_quat_list])
    m = x.T @ x
    w, v = np.linalg.eig(m)
    return v[:, np.argmax(w)]



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

def mnn_mather(desc1, desc2, use_new_method):
    sim = desc1 @ desc2.transpose()
    sim = (sim - np.min(sim))/ (np.max(sim) - np.min(sim))
    if use_new_method == 1:
        dec = 5
    elif use_new_method == 4:
        dec = 0.1
    elif use_new_method == 5:
        dec = 0.3
    elif use_new_method == 6:
        dec = 0.5
    elif use_new_method == 7:
        dec = 1
    elif use_new_method == 8:
        dec = 3
    elif use_new_method == 9:
        dec = 100
    elif use_new_method == 10:
        dec = 10
    elif use_new_method == 11:
        dec = 20
    threshold = np.percentile(sim, 100-dec)
    
    sim[sim < threshold] = 0
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids1 = np.arange(0, sim.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]])
    return matches.transpose()

def matched_points(pts1, pts2, desc1, desc2, opt, args_opt, match='ratio', use_new_method=0):
    if opt[-1] == 'p':
        porce = int(opt[:-1])
        n_key = int(porce/100 * pts1.shape[0])
    else:
        n_key = int(opt)

    s_pts1  = pts1.copy()[:n_key,:]
    s_pts2  = pts2.copy()[:n_key,:]
    s_desc1 = desc1.copy().astype('float32')[:n_key,:]
    s_desc2 = desc2.copy().astype('float32')[:n_key,:]

    if 'orb' in args_opt:
        s_desc1 = s_desc1.astype(np.uint8)
        s_desc2 = s_desc2.astype(np.uint8)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
        matches = bf.match(s_desc1, s_desc2)
    elif use_new_method in [1, 4, 5, 6, 7, 8]:
        matches_idx = mnn_mather(s_desc1, s_desc2, use_new_method)
        matches = [cv2.DMatch(i, j, 0) for i, j in matches_idx]
    elif use_new_method == 2:
        thresh = 0.75
        bf = cv2.BFMatcher(cv2.NORM_L2, False)
        matches1 = bf.knnMatch(s_desc1,s_desc2, k=2)
        matches2 = bf.knnMatch(s_desc2,s_desc1, k=2)
        good1 = []
        for m, n in matches1:
            if m.distance < thresh * n.distance:
                good1.append(m)
        good2 = []
        for m, n in matches2:
            if m.distance < thresh * n.distance:
                good2.append(m)
        good = []
        for m1 in good1:
            for m2 in good2:
                if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
                    good.append(m1)
                    break
        matches = good
    elif use_new_method == 3:
        thresh = 0.75
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(s_desc1, s_desc2, k=2)
        good = []
        for m,n in matches:
            if m.distance < thresh * n.distance:
                good.append(m)
        matches = good
    elif match == 'ratio':
        thresh = 0.75
        bf = cv2.BFMatcher(cv2.NORM_L2, False)
        matches = bf.knnMatch(s_desc1,s_desc2, k=2)
        good = []
        for m,n in matches:
            if m.distance < thresh * n.distance:
                good.append(m)
        matches = good
    else:
        raise ValueError("Invalid matching method specified.")
    


    M = np.zeros((2,len(matches)))
    for ind, match in zip(np.arange(len(matches)),matches):
        M[0,ind] = match.queryIdx
        M[1,ind] = match.trainIdx


    return s_pts1, s_pts2, s_pts1[M[0,:].astype(int),:3], s_pts2[M[1,:].astype(int),:3]




def get_error(x1, x2, Rx, Tx):

    S = computeEssentialMatrixByRANSAC(x1, x2)
    I = S[1]
    I = I.astype(np.int64)

    x1 = x1[I,:]
    x2 = x2[I,:]

    F = calc_ematrix(x1,x2)


    R1,R2,T1,T2 = decomposeE(F)

    R,T = choose_rt(R1,R2,T1,T2,x1,x2)

    R_error = r_error(Rx,R)
    T_error = t_error(Tx,T)

    return R_error, T_error

def get_descriptor(descriptor):
    if descriptor == 'sphorb':
        return 'sphorb', 'erp', 640, 0
    elif descriptor == 'sift':
        return 'sift', 'erp', 512, 0
    elif descriptor == 'tsift':
        return 'sift', 'tangent', 512, 0
    elif descriptor == 'orb':
        return 'orb', 'erp', 512, 0
    elif descriptor == 'torb':
        return 'orb', 'tangent', 512, 0
    elif descriptor == 'spoint':
        return 'superpoint', 'erp', 512, 0
    elif descriptor == 'tspoint':
        return 'superpoint', 'tangent', 512, 0
    elif descriptor == 'alike':
        return 'alike', 'erp', 512, 0
    elif descriptor == 'talike':
        return 'alike', 'tangent', 512, 0
    elif descriptor == 'Proposed':
        return 'superpoint', 'tangent', 512, 1
    elif descriptor == 'Ltspoint':
        return 'superpoint', 'tangent', 512, 2
    elif descriptor == 'Ftspoint':
        return 'superpoint', 'tangent', 512, 3
    elif descriptor == 'Proposed01':
        return 'superpoint', 'tangent', 512, 4
    elif descriptor == 'Proposed03':
        return 'superpoint', 'tangent', 512, 5
    elif descriptor == 'Proposed05':
        return 'superpoint', 'tangent', 512, 6
    elif descriptor == 'Proposed1':
        return 'superpoint', 'tangent', 512, 7
    elif descriptor == 'Proposed3':
        return 'superpoint', 'tangent', 512, 8
    elif descriptor == 'Proposed_un':
        return 'superpoint', 'tangent', 512, 9
    elif descriptor == 'Proposed10':
        return 'superpoint', 'tangent', 512, 10
    elif descriptor == 'Proposed20':
        return 'superpoint', 'tangent', 512, 11


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


def get_kd(array):

    array = np.array(array)
    delimiter = int(array[-1])
    A = array[:-1]
    K = A[:delimiter].reshape(-1,3)
    D = A[delimiter:].reshape(-1,32)
    return K,D

#python3 Accuracy_verification.py --descriptors orb torb sift tsift spoint tspoint sphorb --pose pose1 --solver GSM_SK --inliers 5PA

if __name__ == '__main__':
    main()