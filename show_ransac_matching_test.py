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
    if descriptor[-1] == 'P':
        method_flag = 1
        descriptor = descriptor[:-2]
    else:
        method_flag = 0  
    opt, mode, sphered = get_descriptor(descriptor)
    method_idx = 0.0
    base_order = 0  # Base sphere resolution
    sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
    scale_factor = 1.0  # How much to scale input equirectangular image by
    save_ply = False  # Whether to save the PLY visualizations too
    dim = np.array([2*sphered, sphered])
    img_hw = (512, 1024)

    path_o = args.path + '/O.png'
    path_r = args.path + '/R.png'
    path_o2 = args.path + f'/O2.png'
    path_r2 = args.path + f'/R2.png'
    remap_image(path_o, path_o2)
    remap_image(path_r, path_r2)
    
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
        pts12_, desc12_ = process_image_to_keypoints(path_o2, corners, scale_factor, base_order, sample_order, opt, mode)
        pts22_, desc22_ = process_image_to_keypoints(path_r2, corners, scale_factor, base_order, sample_order, opt, mode)

        if method_flag:
            pts1_ = round_coordinates(pts1_)
            pts2_ = round_coordinates(pts2_)
            pts1_, desc1_ = filter_middle_latitude(pts1_, desc1_, img_hw)
            pts2_, desc2_ = filter_middle_latitude(pts2_, desc2_, img_hw)
            pts12_, desc12_ = filter_middle_latitude(pts12_, desc12_, img_hw)
            pts22_, desc22_ = filter_middle_latitude(pts22_, desc22_, img_hw)

            pts12_ = convert_coordinates_vectorized(pts12_, img_hw)
            pts22_ = convert_coordinates_vectorized(pts22_, img_hw)

            pts1_ = torch.cat((pts1_, pts12_), dim=0)
            desc1_ = torch.cat((desc1_, desc12_), dim=1)
            pts2_ = torch.cat((pts2_, pts22_), dim=0)
            desc2_ = torch.cat((desc2_, desc22_), dim=1)

        


        num_points = args.points
        num_points_1 = num_points
        num_points_2 = num_points
        if args.match == 'MNN':
            num_points_1 = min(num_points_1, 3000)
            num_points_2 = min(num_points_2, 3000)
        pts1, desc1, score1 = sort_key_div(pts1_, desc1_, num_points_1)   
        pts2, desc2, score2 = sort_key_div(pts2_, desc2_, num_points_2)  
        

    else:           
        os.chdir('SPHORB-master/')
        path_o = "."+path_o
        path_r = "."+path_r
        pts1, desc1 = get_kd(sphorb.sphorb(path_o, args.points))
        pts2, desc2 = get_kd(sphorb.sphorb(path_r, args.points))
        pts1[pts1[:,0] > img_o.shape[1], 0] -= img_o.shape[1]
        pts2[pts2[:,0] > img_o.shape[1], 0] -= img_o.shape[1]
        os.chdir('../')



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
            #continue
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



def round_coordinates(tensor):
    coords_int = tensor[:, :2].int()
    other_data = tensor[:, 2:]
    rounded_tensor = torch.cat((coords_int, other_data), dim=1)
    return rounded_tensor


def filter_middle_latitude(pts_, desc_, img_hw, invert_mask=False):
    spherical_coords = equirectangular_to_spherical_coords(img_hw)
    x_indices = pts_[:, 0].long()
    y_indices = pts_[:, 1].long()

    theta_values = spherical_coords[y_indices, x_indices, 0]
    mask = (torch.pi/4 <= theta_values) & (theta_values < 3*torch.pi/4)
    if invert_mask:
        mask = ~mask

    pts = pts_[mask]
    desc = desc_.T[mask].T

    return pts, desc


def equirectangular_to_spherical_coords(img_hw, device='cpu'):
    img_height = img_hw[0]
    img_width = img_hw[1]
    theta = torch.linspace(0, np.pi, img_height, device=device)
    phi = torch.linspace(0, 2 * np.pi, img_width, device=device) 
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="xy")
    return torch.stack([theta_grid, phi_grid, torch.ones_like(theta_grid)], dim=-1)


def spherical_to_cartesian(phi, theta):
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return x, y, z

def rotate_coordinates(x, y, z, angle=np.pi/2):
    xx = x * np.cos(angle) + z * np.sin(angle)
    yy = y
    zz = -x * np.sin(angle) + z * np.cos(angle)
    return xx, yy, zz

def cartesian_to_spherical(x, y, z):
    theta = np.arcsin(z)
    phi = np.arctan2(y, x)
    return phi, theta


def convert_coordinate(input_xy, image_size_hw):
    h, w = image_size_hw
    w_half = w / 2
    h_half = h / 2
    
    phi = (input_xy[0] - w_half) * np.pi * 2 /w
    theta = (input_xy[1] - h_half) * np.pi /h
    
    x, y, z = spherical_to_cartesian(phi, theta)
    xx, yy, zz = rotate_coordinates(x, y, z)
    new_phi, new_theta = cartesian_to_spherical(xx, yy, zz)

    new_y = 2*new_theta * h_half / np.pi + h_half
    new_x = new_phi * w_half / np.pi + w_half
    
    return new_x, new_y


def convert_coordinates_vectorized(tensor, image_size_hw):
    h, w = image_size_hw
    w_half = w / 2
    h_half = h / 2

    x_coords = tensor[:, 0]
    y_coords = tensor[:, 1]

    phi = (x_coords - w_half) * np.pi * 2 / w
    theta = (y_coords - h_half) * np.pi / h
    
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)

    angle = np.pi / 2
    xx = x * np.cos(angle) + z * np.sin(angle)
    yy = y
    zz = -x * np.sin(angle) + z * np.cos(angle)

    new_theta = np.arcsin(zz)
    new_phi = np.arctan2(yy, xx)

    new_y = new_theta * h_half / (np.pi/2) + h_half
    new_x = new_phi * w_half / np.pi + w_half
    
    transformed_tensor = tensor
    transformed_tensor[:, 0] = new_x
    transformed_tensor[:, 1] = new_y
    
    return transformed_tensor



if __name__ == '__main__':
    main()