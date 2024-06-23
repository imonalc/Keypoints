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
from utils.method  import *
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
base_order = 0  # Base sphere resolution
sample_order = 8  # Determines sample resolution (10 = 2048 x 4096) (0, 8): 20*256*256
scale_factor = 1.0  # How much to scale input equirectangular image by


def main():

    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--points', type=int, default = 10000)
    parser.add_argument('--match', default="BF")
    parser.add_argument('--padding_length', default=50)
    parser.add_argument('--solver', default="SK")
    parser.add_argument('--inliers', default="8PA")
    parser.add_argument('--datas'      , nargs='+')
    parser.add_argument('--descriptors', nargs='+')
    args = parser.parse_args()


    DATAS       = args.datas
    DESCRIPTORS = args.descriptors
    padding_length = args.padding_length
    # ----------------------------------------------
    # Parameters
    # ----------------------------------------------

    img_sample = cv2.imread('./data/data_100/Room/0/O.png')
    img_hw = img_sample.shape[:2]


    NUM = 0
    METRICS = np.zeros((len(DESCRIPTORS),2))
    np.random.seed(0)
    data = get_data(DATAS)
    for data in DATAS:
        R_ERROR, T_ERROR, TIMES_MAKEMAP, TIMES_REMAP, TIMES_FEATURE, MATCHING_SCORE, MEAN_MATCHING_ACCURCY, MATCHING_NUM, FP_NUM = [], [], [], [], [], [], [], [], []
        for i in range(len(DESCRIPTORS)):
            R_ERROR.append([])
            T_ERROR.append([])
            TIMES_MAKEMAP.append([])
            TIMES_REMAP.append([])
            TIMES_FEATURE.append([])
            MATCHING_SCORE.append([])
            MEAN_MATCHING_ACCURCY.append([])
            MATCHING_NUM.append([])
            FP_NUM.append([])

        mypath = os.path.join('data/data_100',data)
        paths  = [os.path.join(os.getcwd(),'data/data_100',data,f) for f in listdir('data/data_100/'+data) if isdir(join(mypath, f))]
        NUM = NUM + len(paths)
        std = []

        for path in tqdm(paths):
            path_o = path + '/O.png'
            path_r = path + '/R.png'
            Rx = np.load(path+"/R.npy")
            Tx = np.load(path+"/T.npy")

            for indicador, descriptor in enumerate(DESCRIPTORS):

                try:
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
                        feature_time = t_featurepoint_a - t_featurepoint_b
                        os.chdir('../')
                    else:
                        pts1_, desc1_, make_map_time, remap_time, feature_time = process_image_to_keypoints(path_o, scale_factor, base_order, sample_order, opt, mode, img_hw)
                        pts2_, desc2_, make_map_time, remap_time, feature_time = process_image_to_keypoints(path_r, scale_factor, base_order, sample_order, opt, mode, img_hw)


                    num_points = args.points
                    pts1, desc1, _ = sort_key_div(pts1_, desc1_, num_points)   
                    pts2, desc2, _ = sort_key_div(pts2_, desc2_, num_points)
                    len_pts = (len(pts1) + len(pts2)) / 2

                    if len(pts1.shape) == 1:
                        pts1 = pts1.reshape(1,-1)
                    if len(pts2.shape) == 1:
                        pts2 = pts2.reshape(1,-1)


                    if pts1.shape[0] > 0 or pts2.shape[0] >0:
                        s_pts1, s_pts2, x1_, x2_ = matched_points(pts1, pts2, desc1, desc2, "100p", opt, args.match)
                        x1,x2 = coord_3d(x1_, dim), coord_3d(x2_, dim)
                        s_pts1, s_pts2 = coord_3d(s_pts1, dim), coord_3d(s_pts2, dim)
                        
                        if x1.shape[0] < 8:
                            R_error, T_error = 3.14, 3.14
                        else:
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
                            R1_,R2_,T1_,T2_ = decomposeE(E.T)
                            R_,T_ = choose_rt(R1_,R2_,T1_,T2_,x1,x2)
                            R_error, T_error = r_error(Rx,R_), t_error(Tx,T_)
                            count_inliers = np.sum(inlier_idx == 1)

                        R_ERROR[indicador].append(R_error)
                        T_ERROR[indicador].append(T_error)
                        TIMES_MAKEMAP[indicador].append(make_map_time)
                        TIMES_REMAP[indicador].append(remap_time)
                        TIMES_FEATURE[indicador].append(feature_time)
                        MATCHING_SCORE[indicador].append(count_inliers / len_pts)
                        MEAN_MATCHING_ACCURCY[indicador].append(count_inliers/len(inlier_idx))
                        MATCHING_NUM[indicador].append(count_inliers)
                        FP_NUM[indicador].append(len_pts)

                        METRICS[indicador,:] = METRICS[indicador,:] + [x1.shape[0], (s_pts1.shape[0]+s_pts2.shape[1])/2]

                        std.append(x1.shape[0])
                except:     
                    print("Unexpected error:",indicador, opt)


        for indicador, descriptor in enumerate(DESCRIPTORS):
            base_path = f'results/data_100/FP_{args.points}/values/'+data+'/'+descriptor+'/'+args.match+'_'+args.inliers+'_'+args.solver
            os.system('mkdir -p '+base_path)
            np.savetxt(base_path+'/R_ERRORS.csv',np.array(R_ERROR[indicador]),delimiter=",")
            np.savetxt(base_path+'/T_ERRORS.csv',np.array(T_ERROR[indicador]),delimiter=",")
            np.savetxt(base_path+'/TIMES_MAKEMAP.csv',np.array(TIMES_MAKEMAP[indicador]),delimiter=",")
            np.savetxt(base_path+'/TIMES_REMAP.csv',np.array(TIMES_REMAP[indicador]),delimiter=",")
            np.savetxt(base_path+'/TIMES_FEATURE.csv',np.array(TIMES_FEATURE[indicador]),delimiter=",")
            np.savetxt(base_path+'/MATCHING_SCORE.csv',np.array(MATCHING_SCORE[indicador]),delimiter=",")
            np.savetxt(base_path+'/MEAN_MATCHING_ACCURCY.csv',np.array(MEAN_MATCHING_ACCURCY[indicador]),delimiter=",")
            np.savetxt(base_path+'/MATCHING_NUM.csv',np.array(MATCHING_NUM[indicador]),delimiter=",")
            np.savetxt(base_path+'/FP_NUM.csv',np.array(FP_NUM[indicador]),delimiter=",")

    print('finish')


if __name__ == '__main__':
    main()


