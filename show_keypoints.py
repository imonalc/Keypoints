import sys
import os
sys.path.append(os.getcwd()+'/fivepoint')
import build.fivep as f
import time
import torch
import torch.nn.functional as F
from spherical_distortion.util import *
import matplotlib.pyplot as plt
from skimage import io
import os
import cv2

import sys
import pandas as pd
import numpy as np
import argparse


from utils.coord    import coord_3d
from utils.ransac   import *
from utils.keypoint import *
from utils.metrics  import *
from utils.camera_recovering import *
from utils.matching import *
from utils.spherical_module import * 

from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm

sys.path.append(os.getcwd()+'/SPHORB-master')

import build1.sphorb_cpp as sphorb


def main():

    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--points', type=int, default = 10000)
    parser.add_argument('--descriptor', default = 'sift')
    parser.add_argument('--path', default = "./data/data_100/Room/0")
    args = parser.parse_args()


    print('X')

    descriptor = args.descriptor
    path = args.path
    img_sample = cv2.imread('./data/data_100/Room/0/O.png')
    img_hw = img_sample.shape[:2]
    Y_remap, X_remap = make_image_map(img_hw)
    path_o = path + f'/O.png'
    path_o2 = path + f'/O2.png'
    path_op = path + f'/Op.png'
    path_op2 = path + f'/Op2.png'
    remap_t1 = remap_image(path_o, path_o2, (Y_remap, X_remap))
    method_flag = 0
    if descriptor[-1] == 'P':
        method_flag = 1
        descriptor = descriptor[:-2]
    elif descriptor[-1] == 'p':
        method_flag = 2
        descriptor = descriptor[:-2]
    elif descriptor[-1] == 'a':
        method_flag = 3
        descriptor = descriptor[:-2]
    if method_flag in [2, 3]:
        padding_length = 38
        img_hw_crop = (img_hw[0]//2+padding_length*2+4, img_hw[1]*3//4+padding_length*2+4)
        crop_start_xy = ((img_hw[0]-img_hw_crop[0])//2, (img_hw[1]-img_hw_crop[1])//2)
        img_o = cv2.imread(path_o)
        img_o_cropped = img_o[crop_start_xy[0]:crop_start_xy[0]+img_hw_crop[0], crop_start_xy[1]:crop_start_xy[1]+img_hw_crop[1]]
        cv2.imwrite(path_op, img_o_cropped)
        img_o2 = cv2.imread(path_o2)
        img_o2_cropped = img_o2[crop_start_xy[0]:crop_start_xy[0]+img_hw_crop[0], crop_start_xy[1]:crop_start_xy[1]+img_hw_crop[1]]
        cv2.imwrite(path_op2, img_o2_cropped)

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
        if method_flag == 0:
            t_featurepoint_b = time.perf_counter()
            pts1_, desc1_ = process_image_to_keypoints(path_o, scale_factor, base_order, sample_order, opt, mode)
            t_featurepoint_a = time.perf_counter()
        elif method_flag == 1:
            t_featurepoint_b = time.perf_counter()
            pts1_, desc1_ = process_image_to_keypoints(path_o, scale_factor, base_order, sample_order, opt, mode)
            pts12_, desc12_ = process_image_to_keypoints(path_o2, scale_factor, base_order, sample_order, opt, mode)
            pts12_ = convert_coordinates_vectorized(pts12_, img_hw)
            pts1_, desc1_ = filter_middle_latitude(pts1_, desc1_, img_hw)
            pts12_, desc12_ = filter_middle_latitude(pts12_, desc12_, img_hw, invert_mask=True)
        elif method_flag == 2:
            t_featurepoint_b = time.perf_counter()
            pts1_, desc1_ = process_image_to_keypoints(path_op, scale_factor, base_order, sample_order, opt, mode)
            pts12_, desc12_ = process_image_to_keypoints(path_op2, scale_factor, base_order, sample_order, opt, mode)
            pts1_ = add_offset_to_image(pts1_, crop_start_xy)
            pts12_ = add_offset_to_image(pts12_, crop_start_xy)
            pts12_ = convert_coordinates_vectorized(pts12_, img_hw)
            pts1_, desc1_ = filter_keypoints(pts1_, desc1_, img_hw)
            pts12_, desc12_ = filter_keypoints(pts12_, desc12_, img_hw, invert_mask=True)
        elif method_flag == 3:
            t_featurepoint_b = time.perf_counter()
            pts1_, desc1_ = process_image_to_keypoints(path_op, scale_factor, base_order, sample_order, opt, mode)
            pts12_, desc12_ = process_image_to_keypoints(path_op2, scale_factor, base_order, sample_order, opt, mode)
            pts1_ = add_offset_to_image(pts1_, crop_start_xy)
            pts12_ = add_offset_to_image(pts12_, crop_start_xy)
            pts12_ = convert_coordinates_vectorized(pts12_, img_hw)
            pts1_, desc1_ = filter_keypoints_abridged(pts1_, desc1_, img_hw)
            pts12_, desc12_ = filter_keypoints_abridged(pts12_, desc12_, img_hw, invert_mask=True)
        #if method_flag:
        #    pts1_ = torch.cat((pts1_, pts12_), dim=0)
        #    desc1_ = torch.cat((desc1_, desc12_), dim=1)
        if True:
            pts1_ = pts12_
            desc1_ = desc12_
        pts1, desc1, score1 = sort_key_div(pts1_, desc1_, args.points)
        print(pts1.shape)

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




    fig, ax = plt.subplots(1, 1)

    # Set up the plot
    ax.set_aspect(1, adjustable='box')
    ax.imshow(img)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.plot(pts1_[:, 0], pts1_[:, 1], 'b.', markersize=3.0)
    plt.axis('off')

    plt.show()




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




if __name__ == '__main__':
    main()


