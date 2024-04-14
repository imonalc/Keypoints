import sys
import os
sys.path.append('.'+'/fivepoint')
import build.fivep as f
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
from PIL import Image
import random

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
from torchvision import transforms


sys.path.append('.'+'/SPHORB-master')

import build1.sphorb_cpp as sphorb

#sys.path.append('..')
#from SuperGluePretrainedNetwork.models.matching import Matching

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue

from LightGlue.lightglue import LightGlue, SuperPoint, DISK
from LightGlue.lightglue.utils import load_image, rbd
from LightGlue.lightglue import viz2d

sys.path.append('.'+'/SphereGlue')
from SphereGlue.model.sphereglue import SphereGlue
from SphereGlue.utils.demo_mydataset import MyDataset 



MATCHING_METHODS_LIST = ['BF', 'BF_KNN', 'FLANN_KNN', 'MNN']
MATCHING_CONSTANT_DICT = {
    'BF': [0.75, 0.8],
    'BF_KNN': [0.75, 0.8],
    'FLANN_KNN': [0.75, 0.8],
    'MNN': [0],
}


def main():

    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--points', type=int, default = 10000)
    parser.add_argument('--match', default="BF")
    parser.add_argument('--g_metrics',default="False")
    parser.add_argument('--solver', default="GSM_wRT")
    parser.add_argument('--inliers', default="8PA")
    parser.add_argument('--descriptor', default = 'sift')
    parser.add_argument('--path', default = "./data/data_100/Room/97")
    args = parser.parse_args()


    print('X')

    t0 = time.time()
    descriptor = args.descriptor

    opt, mode, sphered = get_descriptor(descriptor)
    method_idx = 0.0
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
        t_featurepoint_b = time.perf_counter()
        pts1_, desc1_ = process_image_to_keypoints(path_o, corners, scale_factor, base_order, sample_order, opt, mode)
        pts2_, desc2_ = process_image_to_keypoints(path_r, corners, scale_factor, base_order, sample_order, opt, mode)
        t_featurepoint_a = time.perf_counter()
        pts1_, desc1_ = adjust_vertical_intensity(pts1_, desc1_, img_o.shape)
        pts2_, desc2_ = adjust_vertical_intensity(pts2_, desc2_, img_o.shape)
        #pts1, pts2, desc1, desc2, scores1, scores2 = sort_key(pts1_, pts2_, desc1_, desc2_, args.points)
        pts1, pts2, desc1, desc2, scores1, scores2 = sort_key(pts1_, pts2_, desc1_, desc2_, int(pts1_.shape[0]*0.9))
        

    else:           
        os.chdir('SPHORB-master/')
        path_o = "."+path_o
        path_r = "."+path_r
        pts1, desc1 = get_kd(sphorb.sphorb(path_o, args.points))
        pts2, desc2 = get_kd(sphorb.sphorb(path_r, args.points))
        pts1[pts1[:,0] > img_o.shape[1], 0] -= img_o.shape[1]
        pts2[pts2[:,0] > img_o.shape[1], 0] -= img_o.shape[1]
        os.chdir('../')

    #pts1, pts2, desc1, desc2 = mask_cameraman(pts1, pts2, desc1, desc2, img_o.shape)



    if len(pts1.shape) == 1:
        pts1 = pts1.reshape(1,-1)

    t_matching_b = time.perf_counter()
    s_pts1, s_pts2, x1_, x2_ = matched_points(pts1, pts2, desc1, desc2, "100p", opt, match=args.match, constant=method_idx)
    t_matching_a = time.perf_counter()
    
    print(x1_.shape, s_pts1.shape, pts1_.shape)
    x1,x2 = coord_3d(x1_, dim), coord_3d(x2_, dim)
    s_pts1, s_pts2 = coord_3d(s_pts1, dim), coord_3d(s_pts2, dim)

    E, can, inlier_idx = get_cam_pose_by_ransac_GSM_const_wRT(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
    R1_,R2_,T1_,T2_ = decomposeE(E.T)
    R_,T_ = choose_rt(R1_,R2_,T1_,T2_,x1,x2)
    print("True:", sum(inlier_idx), len(inlier_idx), ", ratio:", sum(inlier_idx) / len(inlier_idx))
    print("FP:", "{:.4g}".format((t_featurepoint_a-t_featurepoint_b)/2))
    print("MC:", "{:.4g}".format((t_matching_a-t_matching_b)))

    #print(R_)
    #print(T_)
    
    vis_img = plot_matches(img_o, img_r, s_pts1[:, :2], s_pts2[:, :2], x1_[:, :2], x2_[:, :2], inlier_idx)
    vis_img = cv2.resize(vis_img,dsize=(512,512))
    cv2.imshow("aaa", vis_img)
    c = cv2.waitKey()



def mask_cameraman(pts1, pts2, desc1, desc2, img_o_shape):
    height_threshold = 0.75 * img_o_shape[0]
    cond1_1 = (pts1[:, 1] < height_threshold)
    cond2_1 = (pts2[:, 1] < height_threshold)
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
    cond1_2 = ~(((500 < pts1[:, 0]) &(pts1[:, 0] < 1000))  & (pts1[:, 1] > 360))
    cond1_3 = ~(((800 < pts1[:, 0]) &(pts1[:, 0] < 1000)) & ((240< pts1[:, 1])))
    cond2_2 = ~((pts2[:, 0] < 300) & (pts2[:, 1] > 300))
    cond2_3 = ~(((60 < pts2[:, 0]) &(pts2[:, 0] < 240)) & ((250< pts2[:, 1])))
        # pose5
    #cond1_2 = ~(((400 < pts1[:, 0]) &(pts1[:, 0] < 900))  & (pts1[:, 1] > 360))
    #cond1_3 = ~(((650 < pts1[:, 0]) &(pts1[:, 0] < 850)) & ((240< pts1[:, 1])))
    #cond2_2 = ~((pts2[:, 0] < 400) & (pts2[:, 1] > 400))
    #cond2_3 = ~(((100 < pts2[:, 0]) &(pts2[:, 0] < 350)) & ((250< pts2[:, 1])))
    valid_idx1 = cond1_1 & cond1_2 &cond1_3
    pts1 =  pts1[valid_idx1]
    desc1 = desc1[valid_idx1]
    valid_idx2 = cond2_1 & cond2_2 &cond2_3
    pts2 =  pts2[valid_idx2]
    desc2 = desc2[valid_idx2]

    #valid_idx1 = cond1_1
    #pts1 =  pts1[valid_idx1]
    #desc1 = desc1[valid_idx1]
    #valid_idx2 = cond2_1
    #pts2 =  pts2[valid_idx2]
    #desc2 = desc2[valid_idx2]
    return pts1, pts2, desc1, desc2




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


    if image0.shape[0] > 2000:
        thickness = 10
    if image0.shape[0] > 1000:
        thickness = 5
    else:
        thickness = 2
    
    for kpt0, kpt1, mt in zip(mkpts0, mkpts1, match_true):
        (x0, y0), (x1, y1) = kpt0, kpt1
        if mt == 0 :
            continue
            mcolor = (0, 0, 255) 
        else :
            mcolor = (0, 255, 0)
        cv2.line(out, (x0%W0, y0), (x1%W1, y1),
                 color=mcolor,
                 thickness=thickness,
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


def convert_to_spherical_coordinates(keypoints, image_width, image_height, device="cuda"):
    """
    全天球画像上の2Dキーポイントを単位球面上の3D座標に変換する関数。

    Args:
        keypoints (np.ndarray): 全天球画像上の2Dキーポイント。形状は[N, 2]。
        image_width (int): 全天球画像の幅。
        image_height (int): 全天球画像の高さ。
        device (str): 変換後のテンソルを配置するデバイス（例: 'cpu', 'cuda:0'）。

    Returns:
        torch.Tensor: 単位球面上の3D座標。形状は[N, 3]。
    """
    # 座標を経度と緯度に変換
    longitude = (keypoints[:, 0] / image_width) * 360 - 180
    latitude = (keypoints[:, 1] / image_height) * 180 - 90

    # 経度と緯度をラジアンに変換
    lon_rad = np.deg2rad(longitude)
    lat_rad = np.deg2rad(latitude)

    # 単位球面上の3D座標に変換
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    # 結果をテンソルに変換して、デバイスに移動
    keypoints3D = np.stack([x, y, z], axis=-1)  # [N, 3]の形状
    keypoints3D_tensor = torch.tensor(keypoints3D, dtype=torch.float).to(device)

    return keypoints3D_tensor


    ###  change brightness and contrast
    #alpha = 1  # コントラストの倍率（1より大きい値でコントラストが上がる）
    #beta = 1  # 明るさの調整値（正の値で明るくなる
    #img_r = cv2.convertScaleAbs(img_r, alpha=alpha, beta=beta)
    #mean, stddev = 2, 5
    #gaussian_noise = np.random.normal(mean, stddev, img_r.shape).astype('uint8')
    #img_r = cv2.add(img_r, gaussian_noise)
    #img_r = np.clip(img_r, 0, 255).astype(np.uint8)
    #img_r_pil = Image.fromarray(img_r)
    #img_r_pil.save(args.path + "/R_BCG.png")
    #path_r = args.path + '/R_BCG1.png'
    #img_r = load_torch_img(path_r)[:3, ...].float()
    #img_r = F.interpolate(img_r.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
    #img_r = torch2numpy(img_r.byte())
    #img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

def adjust_vertical_intensity(pts1_, desc1_, img_shape):
    # 画像の高さに関する情報が必要
    img_height = img_shape[0]  # 仮定した値。実際の画像の高さに置き換えてください。

    # 中心のY座標
    center_y = img_height / 2.0

    # 強度の補正
    for i, pt in enumerate(pts1_):
        # 縦方向の位置に基づく補正
        distance_from_center_y = abs(pt[1] - center_y)
        adjustment_factor = np.sqrt(1 - (distance_from_center_y / center_y)**2)
        #adjustment_factor = 1 - (distance_from_center_y / center_y)
        
        # 真ん中の行では1をかける、それ以外は距離に比例して補正
        if distance_from_center_y == center_y:  # 最上行または最下行
            adjustment_factor = 1 - 1
        
        # 強度を補正
        pts1_[i, 2] *= adjustment_factor
    
    return pts1_, desc1_


if __name__ == '__main__':
    main()