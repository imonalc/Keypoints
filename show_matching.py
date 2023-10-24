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
    parser.add_argument('--points', type=int, default = 500)
    parser.add_argument('--match', default="ratio")
    parser.add_argument('--g_metrics',default="False")
    parser.add_argument('--solver', default="None")
    parser.add_argument('--inliers', default="8PA")
    parser.add_argument('--descriptor', default = 'sift')
    parser.add_argument('--path', default = "./data/data_100/Room/0/")
    args = parser.parse_args()


    print('X')

    descriptor = args.descriptor

    opt, mode, sphered, use_our_method = get_descriptor(descriptor)
    base_order = 0  # Base sphere resolution
    sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
    scale_factor = 1.0  # How much to scale input equirectangular image by
    save_ply = False  # Whether to save the PLY visualizations too
    dim = np.array([2*sphered, sphered])

    path_o = args.path + '/O.png'
    path_r = args.path + '/R.png'
    img_o = load_torch_img(path_o)[:3, ...].float()
    img_o = F.interpolate(img_o.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
    img_r = load_torch_img(path_r)[:3, ...].float()
    img_r = F.interpolate(img_r.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)




    img_o = torch2numpy(img_o.byte())
    img_r = torch2numpy(img_r.byte())
    print(img_o.shape)

    img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

    print(path_o)
    if opt != 'sphorb':
        corners = tangent_image_corners(base_order, sample_order)
        pts1, desc1 = process_image_to_keypoints(path_o, corners, scale_factor, base_order, sample_order, opt, mode)
        pts2, desc2 = process_image_to_keypoints(path_r, corners, scale_factor, base_order, sample_order, opt, mode)
        pts1[pts1[:,0] > img_o.shape[1], 0] -= img_o.shape[1]
        pts2[pts2[:,0] > img_o.shape[1], 0] -= img_o.shape[1]
        pts1, pts2, desc1, desc2 = sort_key(pts1, pts2, desc1, desc2, args.points)

    else:
                        
        os.chdir('SPHORB-master/')
        pts1, desc1 = get_kd(sphorb.sphorb(path_o, args.points))
        pts2, desc2 = get_kd(sphorb.sphorb(path_r, args.points))
        os.chdir('../')

    
    if len(pts1.shape) == 1:
        pts1 = pts1.reshape(1,-1)

    print(len(pts1))
    s_pts1, s_pts2, x1, x2 = matched_points(pts1, pts2, desc1, desc2, "100p", opt, args.match, use_new_method=use_our_method)


    print(s_pts1.shape, x1.shape, x2.shape)

    vis_bool = 1
    if vis_bool:
        match_true = np.zeros(x1.shape[0])
        for idx in range(x1.shape[0]):
            match_true[idx] = 1
            vis_img = plot_matches2(img_o, img_r, s_pts1[:, :2], s_pts2[:, :2], x1[:, :2], x2[:, :2], match_true)
            vis_img = cv2.resize(vis_img,dsize=(1600, 400))
            cv2.imshow("aaa", vis_img)
            c = cv2.waitKey()
            match_true[idx] = 0
    else:
        vis_img = plot_matches(img_o, img_r, s_pts1[:, :2], s_pts2[:, :2], x1[:, :2], x2[:, :2])
        cv2.imshow("aaa", vis_img)
        c = cv2.waitKey()


def plot_matches2(image0,
                 image1,
                 kpts0,
                 kpts1,
                 x1,
                 x2,
                 match_true,
                 radius=2,
                 color=(255, 0, 0)):



    out0 = plot_keypoints(image0, kpts0, radius, color)
    out1 = plot_keypoints(image1, kpts1, radius, color)
    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]
    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = out0
    out[:H1, W0:, :] = out1
    mkpts0, mkpts1 = x1, x2
    mkpts0 = np.round(mkpts0).astype(int)
    mkpts1 = np.round(mkpts1).astype(int)
    i = 0
    for kpt0, kpt1 in zip(mkpts0, mkpts1):
        (x0, y0), (x1, y1) = kpt0, kpt1
        if match_true[i]:
            mcolor=(0, 255, 0)
            print(i)
        else:
            i += 1
            continue
            mcolor=(0, 0, 255)
        i += 1
        cv2.line(out, (x0, y0), (x1 + W0, y1),
                     color=mcolor,
                     thickness=5,
                     lineType=cv2.LINE_AA)
    return out



def plot_matches(image0,
                 image1,
                 kpts0,
                 kpts1,
                 x1,
                 x2,
                 radius=2,
                 color=(255, 0, 0)):
    
    match_true = np.zeros(x1.shape[0])
    #match_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,] tspoint
    #match_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, tsift
    #1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
    #1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
    #1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1,
    #]

    #match_true = [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 
    #              0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 
    #              1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 
    #              1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 
    #              1, 
    #]

    #match_true = [1, 1, 1, 1, 1,   1, 0, 1, 1, 1,   1, 1, 0, 1, 0,   1, 1, 1, 0, 1, 
    #              1, 1, 1, 0, 0,   1, 0, 0, 1, 0,   1, 1, 0, 1, 1,   1, 1, 1, 0, 1, 
    #              1, 1, 0, 0, 0,   1, 0, 1, 0, 1,   1, 1, 1, 1, 1,   0, 0, 1, 1, 1, 
    #              1, 1, 0, 1, 0,   1, 0, 1, 0, 0,   0, 0, 0, 0, 0,   1, 0, 0, 1, 1, 
    #              1, 1, 1, 1, 0,   1, 1, 0, 1, 0,   1, 1, 0, 1, 1,   1, 1, 1, 1, 1, 
    #              1, 1, 1, 1, 0,   0, 1, 1, 1, 1,   0, 1, 
    #]
    
    # Proposed
    #match_true = [1, 1, 1, 1, 1,   1, 1, 1, 1, 0,   1, 1, 1, 1, 1,   1, 0, 1, 1, 1, 
    #              1, 1, 1, 1, 1,   1, 1, 1, 1, 1,   1, 1, 1, 0, 1,   1, 1, 1, 1, 1, 
    #              1, 1, 0, 1, 0,   1, 1, 1, 1, 0,   1, 1, 1, 1, 1,   0, 1, 1, 1, 1, 
    #              1, 1, 1, 1, 1,   1, 0, 1, 1, 1,   1, 1, 1, 1, 0,   1, 1, 1, 1, 1, 
    #              1, 0, 1, 1, 1,   1, 1, 1, 1, 1,   1, 1, 1, 1, 1,   0, 1, 1, 1, 1, 
    #              1, 1, 1, 1, 1,   1, 1, 1, 1, 1,   1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 
    #              1, 1, 1, 0, 1,   1, 1, 1, 1, 0,   1, 1, 1, 1, 0,   1, 1, 1, 0, 1, 
    #              1, 1, 1
    #]

    # spoint
    #match_true = [1, 1, 1, 1, 1,   1, 1, 1, 1, 1,   1, 1, 1, 0, 1,   1, 1, 1, 1, 0, 
    #              1, 1, 1, 1, 1,   1, 1, 1, 
    #]

    # LTspoint
    #match_true = [1, 1, 1, 1, 1,   1, 1, 1, 1, 1,  
#
    #]

    ##FFarm
    #Ltspoint
    match_true = [1, 1, 1, 1, 1,   1, 1, 1, 1, 1,   1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1,   1, 1, 1, 1, 1,   1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1,   1, 1, 1, 1, 1,   1, 1, 1, 1, 1,   1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1,   1, 1, 1, 1, 1,   1, 1, 1, 1, 
    ]   



    out0 = plot_keypoints(image0, kpts0, radius, color)
    out1 = plot_keypoints(image1, kpts1, radius, color)

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = max(H0, H1), W0 + W1
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = out0
    out[:H1, W0:, :] = out1

    mkpts0, mkpts1 = x1, x2
    mkpts0 = np.round(mkpts0).astype(int)
    mkpts1 = np.round(mkpts1).astype(int)
    i = -1
    for kpt0, kpt1 in zip(mkpts0, mkpts1):
        (x0, y0), (x1, y1) = kpt0, kpt1
        i += 1
        mcolor=(0, 0, 255)
        if match_true[i] in [1, 2]:
            continue
        cv2.line(out, (x0, y0), (x1 + W0, y1),
                     color=mcolor,
                     thickness=1,
                     lineType=cv2.LINE_AA)
    i = -1
    for kpt0, kpt1 in zip(mkpts0, mkpts1):
        (x0, y0), (x1, y1) = kpt0, kpt1
        mcolor=(0, 255, 0)
        i += 1
        if match_true[i] in [0, 2]:
            continue
        cv2.line(out, (x0, y0), (x1 + W0, y1),
                     color=mcolor,
                     thickness=1,
                     lineType=cv2.LINE_AA)

    return out


def plot_keypoints(image, kpts, radius=2, color=(0, 0, 255)):
    if image.dtype is not np.dtype('uint8'):
        image = image * 255
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    out = np.ascontiguousarray(image.copy())
    kpts = np.round(kpts).astype(int)

    for kpt in kpts:
        x0, y0 = kpt
        cv2.circle(out, (x0, y0), 10, color, -1, lineType=cv2.LINE_4)
    return out




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


def mnn_mather(desc1, desc2, method="mean_std"):
    sim = desc1 @ desc2.transpose()
    if method == "mean_std":
        k = 4
        threshold = sim.mean() + k * sim.std()
    
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
    elif match == 'mnn' or use_new_method == 1:
        matches_idx = mnn_mather(s_desc1, s_desc2)
        matches = [cv2.DMatch(i, j, 0) for i, j in matches_idx]
    elif use_new_method == 3:
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
    elif descriptor == 'csift':
        return 'sift', 'cube', 512, 0
    elif descriptor == 'cpsift':
        return 'sift', 'cubepad', 512, 0
    elif descriptor == 'orb':
        return 'orb', 'erp', 512, 0
    elif descriptor == 'torb':
        return 'orb', 'tangent', 512, 0
    elif descriptor == 'corb':
        return 'orb', 'cube', 512, 0
    elif descriptor == 'cporb':
        return 'orb', 'cubepad', 512, 0
    elif descriptor == 'spoint':
        return 'superpoint', 'erp', 512, 0
    elif descriptor == 'tspoint':
        return 'superpoint', 'tangent', 512, 0
    elif descriptor == 'cspoint':
        return 'superpoint', 'cube', 512, 0
    elif descriptor == 'cpspoint':
        return 'superpoint', 'cubepad', 512, 0
    elif descriptor == 'alike':
        return 'alike', 'erp', 512, 0
    elif descriptor == 'talike':
        return 'alike', 'tangent', 512, 0
    elif descriptor == 'calike':
        return 'alike', 'cube', 512, 0
    elif descriptor == 'cpalike':
        return 'alike', 'cubepad', 512, 0
    elif descriptor == 'Proposed':
        return 'superpoint', 'tangent', 512, 1
    elif descriptor == 'Ltspoint':
        return 'superpoint', 'tangent', 512, 3

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

def get_data(DATAS):
    if len(DATAS) == 1:
        data = DATAS[0]
    elif set(['Urban1','Urban2','Urban3','Urban4']) == set(DATAS):
        data = 'Outdoor'
    elif set(['Realistic','Interior1','Interior2','Room','Classroom']) == set(DATAS):
        data = 'Indoor'
    elif set(['Urban1_R','Urban2_R','Urban3_R','Urban4_R','Realistic_R','Interior1_R','Interior2_R','Room_R','Classroom_R']) == set(DATAS):
        data = 'OnlyRot'
    elif set(['Urban1_T','Urban2_T','Urban3_T','Urban4_T','Realistic_T','Interior1_T','Interior2_T','Room_T','Classroom_T']) == set(DATAS):
        data = 'OnlyTra'
    else:
        data = ''
        for DA in DATAS:
            data+=DA

    return data


def get_kd(array):

    array = np.array(array)
    delimiter = int(array[-1])
    A = array[:-1]
    K = A[:delimiter].reshape(-1,3)
    D = A[delimiter:].reshape(-1,32)
    return K,D



if __name__ == '__main__':
    main()


