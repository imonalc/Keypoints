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

    opt, mode, sphered, method_idx = get_descriptor(descriptor)
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
    path_r = args.path + '/R_BCG.png'
    img_r = load_torch_img(path_r)[:3, ...].float()
    img_r = F.interpolate(img_r.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
    img_r = torch2numpy(img_r.byte())
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)


    ###  add gaussian noise
    #img_r = torch.from_numpy(img_r).float()
    #std_dev = 0.05
    #noise = torch.randn_like(img_r) * std_dev
    #img_r_noisy = img_r + noise / 100
    #img_r_noisy_clamped = torch.clamp(img_r_noisy, 0, 255)
    #img_r = img_r_noisy_clamped.numpy()
    #img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
#

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


    s_pts1, s_pts2, x1_, x2_ = matched_points(pts1, pts2, desc1, desc2, "100p", opt, args.match, use_new_method=method_idx)
    #elif method_idx == 102:
    #    s_pts1, s_pts2, x1_, x2_, _, _ = sphereglue_matching(pts1, pts2, desc1, desc2, scores1, scores2, args.points, device)
    #elif method_idx == 1002:
    #    config = {'K': 2, 
    #              'GNN_layers': ['cross'], 
    #              'match_threshold': 0.2, 
    #              'sinkhorn_iterations': 20, 
    #              'aggr': 'add', 
    #              'knn': 20, 
    #              'max_kpts': args.points,
    #              'descriptor_dim': 256, 
    #              'output_dim': 512
    #    }
    #    
    #    matching_test = SphereGlue(config).to(device)
    #    model_path = './SphereGlue/model_weights/superpoint/autosaved.pt'
    #    ckpt_data = torch.load(model_path)
    #    matching_test.load_state_dict(ckpt_data["MODEL_STATE_DICT"])
    #    matching_test.eval()
    #    x1_, x2_ = [], []
#
    #    keypoints0 = np.delete(pts1, 2, axis=1)
    #    keypoints0_c = convert_to_spherical_coordinates(keypoints0, 1024, 512)
    #    keypoints0_tensor = torch.tensor(keypoints0_c,dtype=torch.float).unsqueeze(0).to(device)
    #    keypoints1 = np.delete(pts2, 2, axis=1)
    #    keypoints1_c = convert_to_spherical_coordinates(keypoints1, 1024, 512)
    #    keypoints1_tensor = torch.tensor(keypoints1_c,dtype=torch.float).unsqueeze(0).to(device)
    #    print("keypoints", type(keypoints0_tensor), keypoints0_tensor.shape)
#
    #    descriptors0_tensor = torch.tensor(desc1,dtype=torch.float).unsqueeze(0).to(device)
    #    descriptors1_tensor = torch.tensor(desc2,dtype=torch.float).unsqueeze(0).to(device)
    #    print("descriptor", type(descriptors0_tensor), descriptors0_tensor.shape)
#
    #    scores0_tensor = scores1.squeeze(1).unsqueeze(0).to(device)
    #    scores1_tensor = scores2.squeeze(1).unsqueeze(0).to(device)
    #    print("score", type(scores0_tensor), scores0_tensor.shape)
#
    #    data = {
    #        "unitCartesian1": keypoints0_tensor,
    #        "h1": descriptors0_tensor,
    #        "scores1": scores0_tensor,
    #        "unitCartesian2": keypoints1_tensor,
    #        "h2": descriptors1_tensor,
    #        "scores2": scores1_tensor,
    #    }
    #    print(111)
    #    pred = matching_test(data)

    #    print(3333)
    #    #print(pred)
#
    #    print("pred_corr", type(pred["matches0"]), pred["matches0"].shape)
    #    print("matching_scores", type(pred["matching_scores0"]), pred["matching_scores0"].shape)
#
    #    #print(pred["matches0"])
#
    #
    #    kpt1 = keypoints0.copy()
    #    kpt1 = np.hstack((kpt1, np.ones((kpt1.shape[0], 1))))
    #    kpt2 = keypoints1.copy()
    #    kpt2 = np.hstack((kpt2, np.ones((kpt2.shape[0], 1))))
    #    matches1 = pred["matches0"].cpu().numpy()
    #    print(matches1[0].shape)
    #    for i in range(len(matches1[0])):
    #        if matches1[0][i] == -1: continue
    #        x1_.append(kpt1[i])
    #        x2_.append(kpt2[matches1[0][i]])
    #    s_pts1, s_pts2 = np.array(kpt1), np.array(kpt2)
    #    x1_, x2_ = np.array(x1_), np.array(x2_)
    #    
    #    print(8888)
    #    print(type(x1_), x1_.shape)
    #    #print(x1_)
    #    #aaa = [np.min(x1_, axis=0), np.max(x1_, axis=0), np.min(x1_, axis=1), np.max(x1_, axis=1)]
    #    #print("x1_", aaa)
#

    #elif method_idx == 101:
    #    matcher = LightGlue(features="superpoint").eval().to(device)
    #    x1_, x2_ = [], []
#
    #    keypoints0 = np.delete(pts1, 2, axis=1)
    #    keypoints0_tensor = torch.tensor(keypoints0,dtype=torch.float).unsqueeze(0).to('cuda:0')
    #    keypoints1 = np.delete(pts2, 2, axis=1)
    #    keypoints1_tensor = torch.tensor(keypoints1,dtype=torch.float).unsqueeze(0).to('cuda:0')
    #    
#
    #    descriptors0_tensor = torch.tensor(desc1,dtype=torch.float).unsqueeze(0).to('cuda:0')
    #    descriptors1_tensor = torch.tensor(desc2,dtype=torch.float).unsqueeze(0).to('cuda:0')
    #    
#
    #    scores0_tensor = scores1.squeeze().to('cuda:0')
    #    scores1_tensor = scores2.squeeze().to('cuda:0')
    #    
#
    #    data = {
    #        "image0":{
    #            "keypoints": keypoints0_tensor,
    #            "keypoint_scores": scores0_tensor,
    #            "descriptors": descriptors0_tensor
    #        },
    #        "image1":{
    #            "keypoints": keypoints1_tensor,
    #            "keypoint_scores": scores1_tensor,
    #            "descriptors": descriptors1_tensor
    #        }
    #    }
    #    print(time.perf_counter())
    #    pred = matcher(data)
    #    matches = pred['matches']
    #    
    #    matches1 = keypoints0[matches[0][:, 0].cpu().numpy(), :] 
    #    matches2 = keypoints1[matches[0][:, 1].cpu().numpy(), :] 
    #    for i in range(len(matches1)):
    #        x1_.append(matches1[i])
    #        x2_.append(matches2[i])
    #    s_pts1, s_pts2 = np.array(keypoints0), np.array(keypoints1)
    #    x1_, x2_ = np.array(x1_), np.array(x2_)
    #    print(time.perf_counter())
    #elif method_idx == 100:
    #    config = {
    #        'descriptor_dim': 256,
    #        'weights': 'indoor',  # 'indoor' または 'outdoor'
    #        'keypoint_encoder': [32, 64, 128, 256],
    #        'GNN_layers': ['self', 'cross'] * 9,
    #        'sinkhorn_iterations': 100,
    #        'match_threshold': 0.2,
    #    }
    #    x1_, x2_ = [], []
    #    matching = Matching(config).eval().to(device)
    #    img_o_tensor = torch.from_numpy(img_o).float()/255.0
    #    img_o_tensor = img_o_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    #    img_r_tensor = torch.from_numpy(img_r).float()/255.0
    #    img_r_tensor = img_r_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    #    img_o_tensor = img_o_tensor[:, 0:1, :, :]
    #    img_r_tensor = img_r_tensor[:, 0:1, :, :]
    #    
    #    
    #    print(1000)
    #    print("image", type(img_o_tensor), img_o_tensor.shape)
#
    #    keypoints0 = np.delete(pts1, 2, axis=1)
    #    keypoints0_tensor = torch.tensor(keypoints0,dtype=torch.float).to(device)
    #    keypoints1 = np.delete(pts2, 2, axis=1)
    #    keypoints1_tensor = torch.tensor(keypoints1,dtype=torch.float).to(device)
    #    print("keypoints", type(keypoints0_tensor), keypoints0_tensor.shape)
#
    #    descriptors0_tensor = torch.tensor(desc1.T,dtype=torch.float).to(device)
    #    descriptors1_tensor = torch.tensor(desc2.T,dtype=torch.float).to(device)
    #    print("descriptor", type(descriptors0_tensor), descriptors0_tensor.shape)
#
    #    scores0_tensor = scores1.squeeze(1).to(device)
    #    scores1_tensor = scores2.squeeze(1).to(device)
    #    print("score", type(scores0_tensor), scores0_tensor.shape)
    #    print(1111)
#
    #    data = {
    #        "keypoints0": [keypoints0_tensor],
    #        "descriptors0": [descriptors0_tensor],
    #        "scores0": (scores0_tensor, ),
    #        "keypoints1": [keypoints1_tensor],
    #        "descriptors1": [descriptors1_tensor],
    #        "scores1": (scores1_tensor, ),
    #        "image0":img_o_tensor,
    #        "image1":img_r_tensor,
    #    }
#
    #    #print(data["keypoints0"][0].device)
    #    #print(data["descriptors0"][0].device)
    #    #print(data["scores0"][0].device)
    #    
    #    pred = matching(data)
    #    #print(pred[0].keys())
#
    #    kpt1 = keypoints0_tensor.cpu().numpy()
    #    #kpt1 = pred[0]["keypoints0"][0].cpu().numpy()
    #    kpt1 = np.hstack((kpt1, np.ones((kpt1.shape[0], 1))))
    #    kpt2 = keypoints1_tensor.cpu().numpy()
    #    #kpt2 = pred[0]["keypoints1"][0].cpu().numpy()
    #    kpt2 = np.hstack((kpt2, np.ones((kpt2.shape[0], 1))))
    #    matches1 = pred[0]["matches0"][0].cpu().numpy()
    #    matches2 = pred[0]["matches1"][0].cpu().numpy()
    #    for i in range(len(matches1)):
    #        if matches1[i] == -1: continue
    #        x1_.append(kpt1[i])
    #        x2_.append(kpt2[matches1[i]])
    #    s_pts1, s_pts2 = np.array(kpt1), np.array(kpt2)
    #    x1_, x2_ = np.array(x1_), np.array(x2_)


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



if __name__ == '__main__':
    main()


