import sys
import os
sys.path.append('.'+'/fivepoint')
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
from utils.matching import *

from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm

sys.path.append('.'+'/SPHORB-master')

import build1.sphorb_cpp as sphorb

#sys.path.append('..')
#from SuperGluePretrainedNetwork.models.matching import Matching

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():

    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--points', type=int, default = 1000)
    parser.add_argument('--match', default="ratio")
    parser.add_argument('--g_metrics',default="False")
    parser.add_argument('--solver', default="GSM_wRT")
    parser.add_argument('--inliers', default="8PA")
    parser.add_argument('--descriptor', default = 'sift')
    parser.add_argument('--path', default = "./data/Farm/new")
    args = parser.parse_args()


    print('X')

    t0 = time.time()
    descriptor = args.descriptor

    opt, mode, sphered, use_our_method = get_descriptor(descriptor)
    base_order = 0  # Base sphere resolution
    sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
    scale_factor = 1.0  # How much to scale input equirectangular image by
    save_ply = False  # Whether to save the PLY visualizations too
    dim = np.array([2*sphered, sphered])

    path_o = args.path + '/O.png'
    path_r = args.path + '/R.png'
    print(path_o)
    img_o = load_torch_img(path_o)[:3, ...].float()
    img_o = F.interpolate(img_o.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
    img_r = load_torch_img(path_r)[:3, ...].float()
    img_r = F.interpolate(img_r.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)

    img_o = torch2numpy(img_o.byte())
    img_r = torch2numpy(img_r.byte())
    print(img_o.shape)

    img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

    if opt != 'sphorb':
        corners = tangent_image_corners(base_order, sample_order)
        pts1_, desc1_ = process_image_to_keypoints(path_o, corners, scale_factor, base_order, sample_order, opt, mode)
        pts2_, desc2_ = process_image_to_keypoints(path_r, corners, scale_factor, base_order, sample_order, opt, mode)
        pts1, pts2, desc1, desc2, scores1, scores2 = sort_key(pts1_, pts2_, desc1_, desc2_, args.points)
        

    else:           
        os.chdir('SPHORB-master/')
        path_o = "."+path_o
        path_r = "."+path_r
        pts1, desc1 = get_kd(sphorb.sphorb(path_o, args.points))
        pts2, desc2 = get_kd(sphorb.sphorb(path_r, args.points))
        pts1[pts1[:,0] > img_o.shape[1], 0] -= img_o.shape[1]
        pts2[pts2[:,0] > img_o.shape[1], 0] -= img_o.shape[1]
        os.chdir('../')

    #pts1, pts2, desc1, desc2, _, _ = sort_key(pts1, pts2, desc1, desc2, args.points)
    #pts1, desc1 = mask_cameraman(pts1, desc1, img_o.shape)
    #pts2, desc2 = mask_cameraman(pts2, desc2, img_o.shape)



    if len(pts1.shape) == 1:
        pts1 = pts1.reshape(1,-1)

    if use_our_method != 12:
        s_pts1, s_pts2, x1_, x2_ = matched_points(pts1, pts2, desc1, desc2, "100p", opt, args.match, use_new_method=use_our_method)
    else:
        config = {
            'superpoint': {
                'max_keypoints': 1000
            },
            'superglue': {
                'weights': "indoor",
            }
        }
        x1_, x2_ = [], []
        matching = Matching(config).eval().to(device)
        img_o_tensor = torch.from_numpy(img_o).float()/255.0
        img_o_tensor = img_o_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        img_r_tensor = torch.from_numpy(img_r).float()/255.0
        img_r_tensor = img_r_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
        img_o_tensor = img_o_tensor[:, 0:1, :, :]
        img_r_tensor = img_r_tensor[:, 0:1, :, :]
        
        
        print(1000)
        print(type(img_o_tensor), img_o_tensor.shape)
        print(type(img_r_tensor), img_r_tensor.shape)

        print(1111)
        keypoints0 = np.delete(pts1, 2, axis=1)
        keypoints0_tensor = torch.tensor(keypoints0)
        keypoints1 = np.delete(pts2, 2, axis=1)
        keypoints1_tensor = torch.tensor(keypoints1)
        print(type(keypoints0_tensor), keypoints0_tensor.shape)
        print(type(keypoints1_tensor), keypoints1_tensor.shape)
        print(2222)

        descriptors0_tensor = torch.tensor(desc1.T)
        descriptors1_tensor = torch.tensor(desc2.T)
        print(type(descriptors0_tensor), descriptors0_tensor.shape)
        print(type(descriptors1_tensor), descriptors1_tensor.shape)
        print(3333)

        scores0_tensor = scores1.squeeze()
        scores1_tensor = scores2.squeeze()
        print(type(scores0_tensor), scores0_tensor.shape)
        print(type(scores1_tensor), scores1_tensor.shape)
        print(4444)

        data_a = {
            "image0":img_o_tensor,
            "image1":img_r_tensor,
        }

        data_b = {
            "keypoints0": [keypoints0_tensor],
            "descriptors0": (descriptors0_tensor),
            "scores0": (scores0_tensor),
            "keypoints1": [keypoints1_tensor],
            "descriptors1": (descriptors1_tensor),
            "scores1": (scores1_tensor),
        }

        #print(data_b)
        
        pred = matching(data_a)

        
        kpt1 = pred["keypoints0"][0].cpu().numpy()
        kpt1 = np.hstack((kpt1, np.ones((kpt1.shape[0], 1))))
        kpt2 = pred["keypoints1"][0].cpu().numpy()
        kpt2 = np.hstack((kpt2, np.ones((kpt2.shape[0], 1))))
        matches1 = pred["matches0"][0].cpu().numpy()
        matches2 = pred["matches1"][0].cpu().numpy()
        for i in range(len(matches1)):
            if matches1[i] == -1: continue
            x1_.append(kpt1[i])
            x2_.append(kpt2[matches1[i]])
        s_pts1, s_pts2 = np.array(kpt1), np.array(kpt2)
        x1_, x2_ = np.array(x1_), np.array(x2_)


    print(x1_.shape, s_pts1.shape)
    x1,x2 = coord_3d(x1_, dim), coord_3d(x2_, dim)
    s_pts1, s_pts2 = coord_3d(s_pts1, dim), coord_3d(s_pts2, dim)

    E, can, inlier_idx = get_cam_pose_by_ransac_GSM_const_wRT(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
    R1_,R2_,T1_,T2_ = decomposeE(E.T)
    R_,T_ = choose_rt(R1_,R2_,T1_,T2_,x1,x2)
    print("True:", sum(inlier_idx), len(inlier_idx), ", ratio:", sum(inlier_idx) / len(inlier_idx))

    print(R_)
    print(T_)
    
    vis_img = plot_matches(img_o, img_r, s_pts1[:, :2], s_pts2[:, :2], x1_[:, :2], x2_[:, :2], inlier_idx)
    vis_img = cv2.resize(vis_img,dsize=(512,512))
    cv2.imshow("aaa", vis_img)
    c = cv2.waitKey()



def mask_cameraman(pts1, desc1, img_o_shape):
    height_threshold = 0.75 * img_o_shape[0]
    cond1_1 = (pts1[:, 1] < height_threshold)
    #cond2_1 = (pts2[:, 1] < height_threshold)
        # pose1
    # cond1_2 = ~(((400 < pts1[:, 0]) &(pts1[:, 0] < 840))  & (pts1[:, 1] > 400))
    # cond1_3 = ~(((680 < pts1[:, 0]) &(pts1[:, 0] < 800)) & ((220< pts1[:, 1])))
    # cond2_2 = ~((pts2[:, 0] < 400) & (pts2[:, 1] > 350))
    # cond2_3 = ~(((220 < pts2[:, 0]) &(pts2[:, 0] < 320)) & ((250< pts2[:, 1])))
        # pose2
    # cond1_2 = ~(((500 < pts1[:, 0]) &(pts1[:, 0] < 1000))  & (pts1[:, 1] > 360))
    # cond1_3 = ~(((800 < pts1[:, 0]) &(pts1[:, 0] < 1000)) & ((220< pts1[:, 1])))
    # cond2_2 = ~((pts2[:, 0] < 400) & (pts2[:, 1] > 350))
    # cond2_3 = ~(((100 < pts2[:, 0]) &(pts2[:, 0] < 250)) & ((250< pts2[:, 1])))
        # pose3
    #cond1_2 = ~(((200 < pts1[:, 0]) &(pts1[:, 0] < 700))  & (pts1[:, 1] > 360))
    #cond1_3 = ~(((500 < pts1[:, 0]) &(pts1[:, 0] < 750)) & ((240< pts1[:, 1])))
    #cond2_2 = ~((pts2[:, 0] < 400) & (pts2[:, 1] > 300))
    #cond2_3 = ~(((200 < pts2[:, 0]) &(pts2[:, 0] < 400)) & ((250< pts2[:, 1])))
        # pose4
    #cond1_2 = ~(((500 < pts1[:, 0]) &(pts1[:, 0] < 1000))  & (pts1[:, 1] > 360))
    #cond1_3 = ~(((800 < pts1[:, 0]) &(pts1[:, 0] < 1000)) & ((240< pts1[:, 1])))
    #cond2_2 = ~((pts2[:, 0] < 300) & (pts2[:, 1] > 300))
    #cond2_3 = ~(((60 < pts2[:, 0]) &(pts2[:, 0] < 240)) & ((250< pts2[:, 1])))
        # pose5
    #cond1_2 = ~(((400 < pts1[:, 0]) &(pts1[:, 0] < 900))  & (pts1[:, 1] > 360))
    #cond1_3 = ~(((650 < pts1[:, 0]) &(pts1[:, 0] < 850)) & ((240< pts1[:, 1])))
    #cond2_2 = ~((pts2[:, 0] < 400) & (pts2[:, 1] > 400))
    #cond2_3 = ~(((100 < pts2[:, 0]) &(pts2[:, 0] < 350)) & ((250< pts2[:, 1])))
    #valid_idx1 = cond1_1 & cond1_2 &cond1_3
    #pts1 =  pts1[valid_idx1]
    #desc1 = desc1[valid_idx1]
    #valid_idx2 = cond2_1 & cond2_2 &cond2_3
    #pts2 =  pts2[valid_idx2]
    #desc2 = desc2[valid_idx2]

    #valid_idx1 = cond1_1
    #pts1 =  pts1[valid_idx1]
    #desc1 = desc1[valid_idx1]
    #valid_idx2 = cond2_1
    #pts2 =  pts2[valid_idx2]
    #desc2 = desc2[valid_idx2]
    return pts1, desc1




def plot_matches(image0,
                 image1,
                 kpts0,
                 kpts1,
                 x1,
                 x2,
                 inlier_idx,
                 radius=2):
    
    # Convert the bool list of inlier_idx to integer values.
    match_true = np.array(inlier_idx, dtype=int)
  
    out0 = plot_keypoints(image0, kpts0, radius, (255, 0, 0))
    out1 = plot_keypoints(image1, kpts1, radius, (255, 0, 0))

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = H0 + H1, max(W0, W1)
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = out0
    out[H0:H0+H1, :W1, :] = out1

    mkpts0, mkpts1 = x1, x2
    mkpts0 = np.round(mkpts0).astype(int)
    mkpts1 = np.round(mkpts1).astype(int)

    mkpts1[:, 1] += H0
    
    for kpt0, kpt1, mt in zip(mkpts0, mkpts1, match_true):
        (x0, y0), (x1, y1) = kpt0, kpt1
        mcolor = (0, 0, 255) if mt == 0 else (0, 255, 0)  # Red for outliers, Green for inliers
        cv2.line(out, (x0, y0), (x1, y1),
                 color=mcolor,
                 thickness=2,
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
        cv2.circle(out, (x0, y0), 4, color, -1, lineType=cv2.LINE_4)
    return out



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
    elif use_new_method == 1:
        matches_idx = mnn_matcher(s_desc1, s_desc2, use_new_method)
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
    elif descriptor == "glue":
        return "superpoint", "erp", 512, 12






if __name__ == '__main__':
    main()


