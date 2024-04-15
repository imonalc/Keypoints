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
    parser.add_argument('--points', type=int, default = 10000)
    parser.add_argument('--descriptor', default = 'sift')
    parser.add_argument('--path', default = "./data/data_100/Room/0/")
    args = parser.parse_args()


    print('X')

    descriptor = args.descriptor

    opt, mode, sphered = get_descriptor(descriptor)

    base_order = 1  # Base sphere resolution
    sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
    scale_factor = 1.0  # How much to scale input equirectangular image by
    save_ply = False  # Whether to save the PLY visualizations too
    dim = np.array([2*sphered, sphered])

    path_o = args.path + '/O.png'
    print(path_o)
    if opt != 'sphorb':

        corners = tangent_image_corners(base_order, sample_order)

        pts1, desc1 = process_image_to_keypoints(path_o, corners, scale_factor, base_order, sample_order, opt, mode)
        #pts1, desc1 = sort_key(pts1, desc1, args.points)

    else:
                        
        os.chdir('SPHORB-master/')

        path_o = "."+path_o

        pts1, desc1 = get_kd(sphorb.sphorb(path_o, args.points))

        pts1[:,0] = pts1[:,0]/640
        pts1[:,0]*=512

        pts1[:,1] = pts1[:,1]/1280
        pts1[:,1]*=1024


    if len(pts1.shape) == 1:
        pts1 = pts1.reshape(1,-1)
        

    img = load_torch_img(path_o)[:3, ...].float()
    img = F.interpolate(img.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)


    img = torch2numpy(img.byte())
    print(img.shape)

    test = 1
    if test:
        pts1 = round_coordinates(pts1)      
        pts1, desc1 = filter_middle_latitude(pts1, desc1, (512, 1024), invert_mask=True)



    fig, ax = plt.subplots(1, 1)

    # Set up the plot
    ax.set_aspect(1, adjustable='box')
    ax.imshow(img)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.plot(pts1[:, 0], pts1[:, 1], 'b.', markersize=3.0)
    plt.axis('off')

    plt.show()


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
    theta = torch.linspace(0, np.pi, img_height, device=device)  # 0からπ
    phi = torch.linspace(0, 2 * np.pi, img_width, device=device)  # 0から2π
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="xy")
    return torch.stack([theta_grid, phi_grid, torch.ones_like(theta_grid)], dim=-1)


def round_coordinates(tensor):
    coords_int = tensor[:, :2].int()
    other_data = tensor[:, 2:]
    rounded_tensor = torch.cat((coords_int, other_data), dim=1)
    return rounded_tensor



def sort_key(pts1, desc1, points):

    ind1 = np.argsort(pts1[:,2].numpy(),axis = 0)[::-1]

    max1 = np.min([points,ind1.shape[0]])

    ind1 = ind1[:max1]

    pts1 = pts1[ind1.copy(),:]

    desc1 = desc1[:,ind1.copy()]

    pts1 = np.concatenate((pts1[:,:2], np.ones((pts1.shape[0],1))), axis = 1 )

    desc1 = np.transpose(desc1,[1,0]).numpy()

    return pts1, desc1

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
        return 'sphorb', 'erp', 640
    elif descriptor == 'sift':
        return 'sift', 'erp', 512
    elif descriptor == 'tsift':
        return 'sift', 'tangent', 512
    elif descriptor == 'csift':
        return 'sift', 'cube', 512
    elif descriptor == 'orb':
        return 'orb', 'erp', 512
    elif descriptor == 'torb':
        return 'orb', 'tangent', 512
    elif descriptor == 'corb':
        return 'orb', 'cube', 512
    elif descriptor == 'spoint':
        return 'superpoint', 'erp', 512
    elif descriptor == 'tspoint':
        return 'superpoint', 'tangent', 512
    elif descriptor == 'cspoint':
        return 'superpoint', 'cube', 512
    elif descriptor == 'alike':
        return 'alike', 'erp', 512
    elif descriptor == 'talike':
        return 'alike', 'tangent', 512
    elif descriptor == 'calike':
        return 'alike', 'cube', 512


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


