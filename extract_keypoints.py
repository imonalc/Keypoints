import sys
import os
sys.path.append(os.getcwd()+'/fivepoint')
import fivepoint.build.fivep as f

import time
from spherical_distortion.util import *

import torch
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

from PIL import Image
import random

sys.path.append(os.getcwd()+'/SPHORB-master')

import build1.sphorb_cpp as sphorb
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def main():

    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--points', type=int, default = 10000)
    parser.add_argument('--match', default="BF")
    parser.add_argument('--solver', default="SK")
    parser.add_argument('--inliers', default="8PA")
    parser.add_argument('--datas'      , nargs='+')
    parser.add_argument('--descriptors', nargs='+')
    args = parser.parse_args()


    DATAS       = args.datas
    DESCRIPTORS = args.descriptors
    # ----------------------------------------------
    # Parameters
    # ----------------------------------------------

    img_sample = cv2.imread('./data/data_100/Room/0/O.png')
    img_hw = img_sample.shape[:2]
    Y_remap, X_remap = make_image_map(img_hw)


    NUM = 0
    METRICS = np.zeros((len(DESCRIPTORS),2))
    np.random.seed(0)
    data = get_data(DATAS)
    for data in DATAS:
        R_ERROR, T_ERROR, TIMES_FP, TIMES_MC, TIMES_PE, MATCHING_SCORE, MEAN_MATCHING_ACCURCY, MATCHING_NUM, FP_NUM = [], [], [], [], [], [], [], [], []
        for i in range(len(DESCRIPTORS)):
            R_ERROR.append([])
            T_ERROR.append([])
            TIMES_FP.append([])
            TIMES_MC.append([])
            TIMES_PE.append([])
            MATCHING_SCORE.append([])
            MEAN_MATCHING_ACCURCY.append([])
            MATCHING_NUM.append([])
            FP_NUM.append([])

        mypath = os.path.join('data/data_100',data)
        paths  = [os.path.join(os.getcwd(),'data/data_100',data,f) for f in listdir('data/data_100/'+data) if isdir(join(mypath, f))]
        NUM = NUM + len(paths)
        std = []

        for path in tqdm(paths):
            method_idx = 0
            base_order = 0  # Base sphere resolution
            sample_order = 8  # Determines sample resolution (10 = 2048 x 4096) (0, 8): 20*256*256
            scale_factor = 1.0  # How much to scale input equirectangular image by

            img_hw = (512, 1024)
            path_o = path + f'/O.png'
            path_r = path + f'/R.png'
            path_o2 = path + f'/O2.png'
            path_r2 = path + f'/R2.png'
            path_op = path + f'/Op.png'
            path_rp = path + f'/Rp.png'
            path_op2 = path + f'/Op2.png'
            path_rp2 = path + f'/Rp2.png'
            remap_t1 = remap_image(path_o, path_o2, (Y_remap, X_remap))
            remap_t2 = remap_image(path_r, path_r2, (Y_remap, X_remap))
            for indicador, descriptor in enumerate(DESCRIPTORS):

                try:
                    if descriptor[-1] == 'P':
                        method_flag = 1
                        descriptor = descriptor[:-2]
                    elif descriptor[-1] == 'p':
                        method_flag = 2
                        descriptor = descriptor[:-2]
                    elif descriptor[-1] == 'a':
                        method_flag = 3
                        descriptor = descriptor[:-2]
                    else:
                        method_flag = 0
                    
                    if method_flag in [2, 3]:
                        padding_length = 38
                        img_hw_crop = (img_hw[0]//2+padding_length*2+2, img_hw[1]*3//4+padding_length*2+2)
                        crop_start_xy = ((img_hw[0]-img_hw_crop[0])//2 - 1, (img_hw[1]-img_hw_crop[1])//2 - 1)
                        proposed_image_mapping(path_o, path_r, path_o2, path_r2, path_op, path_rp, path_op2, path_rp2, img_hw, crop_start_xy, img_hw_crop)

                    
                    opt, mode, sphered = get_descriptor(descriptor)
                    dim = np.array([2*sphered, sphered])


                    if opt == 'sphorb':
                        os.chdir('SPHORB-master/')
                        t_featurepoint_b = time.perf_counter()
                        pts1, desc1 = get_kd(sphorb.sphorb(path_o, args.points))
                        pts2, desc2 = get_kd(sphorb.sphorb(path_r, args.points))
                        pts1_, desc1_ = convert_sphorb(pts1, desc1)
                        pts2_, desc2_ = convert_sphorb(pts2, desc2)
                        t_featurepoint_a = time.perf_counter()
                        os.chdir('../')

                    else:
                        corners = tangent_image_corners(base_order, sample_order)
                        t_featurepoint_b = time.perf_counter()
                        if method_flag == 0:
                            pts1_, desc1_ = process_image_to_keypoints(path_o, scale_factor, base_order, sample_order, opt, mode)
                            pts2_, desc2_ = process_image_to_keypoints(path_r, scale_factor, base_order, sample_order, opt, mode)
                            t_featurepoint_a = time.perf_counter()
                        elif method_flag == 1:
                            pts1_, desc1_, pts2_, desc2_, pts12_, desc12_, pts22_, desc22_ = method_P(path_o, path_r, path_o2, path_r2, args, img_hw, scale_factor, base_order, sample_order, opt, mode)
                        elif method_flag == 2:
                            pts1_, desc1_, pts2_, desc2_, pts12_, desc12_, pts22_, desc22_ = method_p(path_op, path_rp, path_op2, path_rp2, args, img_hw, crop_start_xy, scale_factor, base_order, sample_order, opt, mode)
                        elif method_flag == 3:
                            pts1_, desc1_, pts2_, desc2_, pts12_, desc12_, pts22_, desc22_ = method_a(path_op, path_rp, path_op2, path_rp2, args, img_hw, crop_start_xy, scale_factor, base_order, sample_order, opt, mode)
                        

                        if method_flag:
                            pts1_ = torch.cat((pts1_, pts12_), dim=0)
                            desc1_ = torch.cat((desc1_, desc12_), dim=1)
                            pts2_ = torch.cat((pts2_, pts22_), dim=0)
                            desc2_ = torch.cat((desc2_, desc22_), dim=1)
                            t_featurepoint_a = time.perf_counter()+remap_t1+remap_t2


                    num_points = args.points
                    num_points_1 = num_points
                    num_points_2 = num_points
                    if args.match == 'MNN':
                        num_points_1 = min(num_points_1, 3000)
                        num_points_2 = min(num_points_2, 3000)
                    pts1, desc1, score1 = sort_key_div(pts1_, desc1_, num_points_1)   
                    pts2, desc2, score2 = sort_key_div(pts2_, desc2_, num_points_2)

                    if len(pts1.shape) == 1:
                        pts1 = pts1.reshape(1,-1)
                    if len(pts2.shape) == 1:
                        pts2 = pts2.reshape(1,-1)
                        
                    Rx = np.load(path+"/R.npy")
                    Tx = np.load(path+"/T.npy")


                    len_pts = (len(pts1) + len(pts2)) / 2

                    if pts1.shape[0] > 0 or pts2.shape[0] >0:
                        t_matching_b = time.perf_counter()
                        s_pts1, s_pts2, x1_, x2_ = matched_points(pts1, pts2, desc1, desc2, "100p", opt, args.match, method_idx)
                        t_matching_a = time.perf_counter()
                        x1,x2 = coord_3d(x1_, dim), coord_3d(x2_, dim)

                        s_pts1, s_pts2 = coord_3d(s_pts1, dim), coord_3d(s_pts2, dim)
                        
                        if x1.shape[0] < 8:
                            R_error, T_error = 3.14, 3.14
                        else:
                            t_poseestimate_b = time.perf_counter()
                            if args.solver   == 'None':
                                E, cam, inlier_idx = get_cam_pose_by_ransac(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                            elif args.solver == 'SK':
                                E, can, inlier_idx = get_cam_pose_by_ransac_opt_SK(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                            elif args.solver == 'GSM':
                                E, can, inlier_idx = get_cam_pose_by_ransac_GSM(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                            elif args.solver == 'GSM_wRT':
                                E, can, inlier_idx = get_cam_pose_by_ransac_GSM_const_wRT(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                            elif args.solver == 'GSM_SK':
                                E, can, inlier_idx = get_cam_pose_by_ransac_GSM_const_wSK(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
                            t_poseestimate_a = time.perf_counter()
                            R1_,R2_,T1_,T2_ = decomposeE(E.T)
                            R_,T_ = choose_rt(R1_,R2_,T1_,T2_,x1,x2)
                            R_error, T_error = r_error(Rx,R_), t_error(Tx,T_)
                            count_inliers = np.sum(inlier_idx == 1)

                        R_ERROR[indicador].append(R_error)
                        T_ERROR[indicador].append(T_error)
                        TIMES_FP[indicador].append((t_featurepoint_a-t_featurepoint_b)/2)
                        TIMES_MC[indicador].append(t_matching_a-t_matching_b)
                        TIMES_PE[indicador].append(t_poseestimate_a-t_poseestimate_b)
                        MATCHING_SCORE[indicador].append(count_inliers / len_pts)
                        MEAN_MATCHING_ACCURCY[indicador].append(count_inliers/len(inlier_idx))
                        MATCHING_NUM[indicador].append(count_inliers)
                        FP_NUM[indicador].append(len_pts)

                        METRICS[indicador,:] = METRICS[indicador,:] + [x1.shape[0], (s_pts1.shape[0]+s_pts2.shape[1])/2]

                        std.append(x1.shape[0])
                except:     
                    print("Unexpected error:",indicador, opt, method_idx)


        for indicador, descriptor in enumerate(DESCRIPTORS):
            base_path = f'results/data_100/FP_{args.points}/values/'+data+'/'+descriptor+'/'+args.match+'_'+args.inliers+'_'+args.solver
            os.system('mkdir -p '+base_path)
            np.savetxt(base_path+'/R_ERRORS.csv',np.array(R_ERROR[indicador]),delimiter=",")
            np.savetxt(base_path+'/T_ERRORS.csv',np.array(T_ERROR[indicador]),delimiter=",")
            np.savetxt(base_path+'/TIMES_FP.csv',np.array(TIMES_FP[indicador]),delimiter=",")
            np.savetxt(base_path+'/TIMES_MC.csv',np.array(TIMES_MC[indicador]),delimiter=",")
            np.savetxt(base_path+'/MATCHING_SCORE.csv',np.array(MATCHING_SCORE[indicador]),delimiter=",")
            np.savetxt(base_path+'/MEAN_MATCHING_ACCURCY.csv',np.array(MEAN_MATCHING_ACCURCY[indicador]),delimiter=",")
            np.savetxt(base_path+'/MATCHING_NUM.csv',np.array(MATCHING_NUM[indicador]),delimiter=",")
            np.savetxt(base_path+'/FP_NUM.csv',np.array(FP_NUM[indicador]),delimiter=",")

    print('finish')



def proposed_image_mapping(path_o, path_r, path_o2, path_r2, path_op, path_rp, path_op2, path_rp2, img_hw, crop_start_xy, img_hw_crop):
    img_o = cv2.imread(path_o)
    img_r = cv2.imread(path_r)
    img_o_cropped = img_o[crop_start_xy[0]:crop_start_xy[0]+img_hw_crop[0], crop_start_xy[1]:crop_start_xy[1]+img_hw_crop[1]]
    img_r_cropped = img_r[crop_start_xy[0]:crop_start_xy[0]+img_hw_crop[0], crop_start_xy[1]:crop_start_xy[1]+img_hw_crop[1]]
    cv2.imwrite(path_op, img_o_cropped)
    cv2.imwrite(path_rp, img_r_cropped)
    img_o2 = cv2.imread(path_o2)
    img_r2 = cv2.imread(path_r2)
    img_o2_cropped = img_o2[crop_start_xy[0]:crop_start_xy[0]+img_hw_crop[0], crop_start_xy[1]:crop_start_xy[1]+img_hw_crop[1]]
    img_r2_cropped = img_r2[crop_start_xy[0]:crop_start_xy[0]+img_hw_crop[0], crop_start_xy[1]:crop_start_xy[1]+img_hw_crop[1]]
    cv2.imwrite(path_op2, img_o2_cropped)
    cv2.imwrite(path_rp2, img_r2_cropped)


def convert_sphorb(pts, desc):
    pts_new = np.hstack((pts, pts[:, 2:3]))
    pts_tensor = torch.tensor(pts_new)
    desc_tensor = torch.tensor(desc.T)
    
    return pts_tensor, desc_tensor


def method_P(path_o, path_r, path_o2, path_r2, args, img_hw, scale_factor, base_order, sample_order, opt, mode):
    pts1_, desc1_ = process_image_to_keypoints(path_o, scale_factor, base_order, sample_order, opt, mode)
    pts2_, desc2_ = process_image_to_keypoints(path_r, scale_factor, base_order, sample_order, opt, mode)
    pts12_, desc12_ = process_image_to_keypoints(path_o2, scale_factor, base_order, sample_order, opt, mode)
    pts22_, desc22_ = process_image_to_keypoints(path_r2, scale_factor, base_order, sample_order, opt, mode)
    pts12_ = convert_coordinates_vectorized(pts12_, img_hw)
    pts22_ = convert_coordinates_vectorized(pts22_, img_hw)
    pts1_, desc1_ = filter_middle_latitude(pts1_, desc1_, img_hw)
    pts2_, desc2_ = filter_middle_latitude(pts2_, desc2_, img_hw)
    pts12_, desc12_ = filter_middle_latitude(pts12_, desc12_, img_hw, invert_mask=True)
    pts22_, desc22_ = filter_middle_latitude(pts22_, desc22_, img_hw, invert_mask=True)

    return pts1_, desc1_, pts2_, desc2_, pts12_, desc12_, pts22_, desc22_



def method_p(path_op, path_rp, path_op2, path_rp2, args, img_hw, crop_start_xy, scale_factor, base_order, sample_order, opt, mode):
    pts1_, desc1_ = process_image_to_keypoints(path_op, scale_factor, base_order, sample_order, opt, mode)
    pts2_, desc2_ = process_image_to_keypoints(path_rp, scale_factor, base_order, sample_order, opt, mode)
    pts12_, desc12_ = process_image_to_keypoints(path_op2, scale_factor, base_order, sample_order, opt, mode)
    pts22_, desc22_ = process_image_to_keypoints(path_rp2, scale_factor, base_order, sample_order, opt, mode)
    pts1_ = add_offset_to_image(pts1_, crop_start_xy)
    pts2_ = add_offset_to_image(pts2_, crop_start_xy)
    pts12_ = add_offset_to_image(pts12_, crop_start_xy)
    pts22_ = add_offset_to_image(pts22_, crop_start_xy)
    pts12_ = convert_coordinates_vectorized(pts12_, img_hw)
    pts22_ = convert_coordinates_vectorized(pts22_, img_hw)
    pts1_, desc1_ = filter_keypoints(pts1_, desc1_, img_hw)
    pts2_, desc2_ = filter_keypoints(pts2_, desc2_, img_hw)
    pts12_, desc12_ = filter_keypoints(pts12_, desc12_, img_hw, invert_mask=True)
    pts22_, desc22_ = filter_keypoints(pts22_, desc22_, img_hw, invert_mask=True)

    return pts1_, desc1_, pts2_, desc2_, pts12_, desc12_, pts22_, desc22_



def method_a(path_op, path_rp, path_op2, path_rp2, args, img_hw, crop_start_xy, scale_factor, base_order, sample_order, opt, mode):
    pts1_, desc1_ = process_image_to_keypoints(path_op, scale_factor, base_order, sample_order, opt, mode)
    pts2_, desc2_ = process_image_to_keypoints(path_rp, scale_factor, base_order, sample_order, opt, mode)
    pts12_, desc12_ = process_image_to_keypoints(path_op2, scale_factor, base_order, sample_order, opt, mode)
    pts22_, desc22_ = process_image_to_keypoints(path_rp2, scale_factor, base_order, sample_order, opt, mode)
    pts1_ = add_offset_to_image(pts1_, crop_start_xy)
    pts2_ = add_offset_to_image(pts2_, crop_start_xy)
    pts12_ = add_offset_to_image(pts12_, crop_start_xy)
    pts22_ = add_offset_to_image(pts22_, crop_start_xy)
    pts12_ = convert_coordinates_vectorized(pts12_, img_hw)
    pts22_ = convert_coordinates_vectorized(pts22_, img_hw)
    pts1_, desc1_ = filter_keypoints_abridged(pts1_, desc1_, img_hw)
    pts2_, desc2_ = filter_keypoints_abridged(pts2_, desc2_, img_hw)
    pts12_, desc12_ = filter_keypoints_abridged(pts12_, desc12_, img_hw, invert_mask=True)
    pts22_, desc22_ = filter_keypoints_abridged(pts22_, desc22_, img_hw, invert_mask=True)

    return pts1_, desc1_, pts2_, desc2_, pts12_, desc12_, pts22_, desc22_


def sort_key_div_torch(pts1, desc1, points):
    ind1 = pts1[:, 2].argsort(descending=True)
    max1 = min(points, ind1.shape[0])
    ind1 = ind1[:max1]
    pts1 = pts1[ind1]
    desc1 = desc1[:, ind1]
    pts1[:, 2] = 1

    return pts1, desc1


if __name__ == '__main__':
    main()


