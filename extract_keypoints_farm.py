import sys
import os
sys.path.append(os.getcwd()+'/fivepoint')
import fivepoint.build.fivep as f

import time
from spherical_distortion.util import *

import pandas as pd
import numpy as np
import argparse

from scipy.spatial.transform import Rotation

from utils.coord    import coord_3d
from utils.ransac   import *
from utils.keypoint import *
from utils.metrics  import *
from utils.camera_recovering import *
from utils.matching import *

from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm

sys.path.append(os.getcwd()+'/SPHORB-master')

import build1.sphorb_cpp as sphorb



def main():

    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--points', type=int, default = 10000)
    parser.add_argument('--match', default="BF")
    parser.add_argument('--g_metrics',default="False")
    parser.add_argument('--solver', default="GSM_wRT")
    parser.add_argument('--inliers', default="8PA")
    parser.add_argument('--pose'      , default="pose_test")
    parser.add_argument('--scene'      , default="Room")
    parser.add_argument('--descriptors', nargs='+')
    args = parser.parse_args()

    DESCRIPTORS = args.descriptors
    # ----------------------------------------------
    # Parameters
    # ----------------------------------------------

    NUM = 0
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


    METRICS = np.zeros((len(DESCRIPTORS),2))

    pose = args.pose
    scene = args.scene
    path_true_value = os.path.join(f"./data/data_real/Calibration/", pose)
    Rx = np.load(path_true_value + "/R.npy")
    Tx = np.load(path_true_value + "/T.npy")

    mypath = os.path.join(f'./data/data_real/{scene}/',pose)
    paths  = [os.path.join(f'./data/data_real/{scene}/',pose,f) for f in listdir(f'./data/data_real/{scene}/'+pose) if isdir(join(mypath, f))]
    print(os.path.join(f'./data/data_real/{scene}/',pose))
    NUM = NUM + len(paths)
    #print(paths)

    std = []
    for path in tqdm(paths):

        for indicador, descriptor in enumerate(DESCRIPTORS):

            try:
                if descriptor[:-1] == "Proposed":
                    opt, mode, sphered = get_descriptor("spoint")
                else:
                    opt, mode, sphered = get_descriptor(descriptor)
                method_idx = 0
                base_order = 0  # Base sphere resolution
                sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
                scale_factor = 1.0  # How much to scale input equirectangular image by
                save_ply = False  # Whether to save the PLY visualizations too
                dim = np.array([2*sphered, sphered])
                path_o = path + f'/O.png'
                path_r = path + f'/R.png'

                if opt == 'sphorb':
                    os.chdir('SPHORB-master/')
                    t_featurepoint_b = time.perf_counter()
                    pts1, desc1 = get_kd(sphorb.sphorb(path_o, args.points))
                    pts2, desc2 = get_kd(sphorb.sphorb(path_r, args.points))
                    t_featurepoint_a = time.perf_counter()
                    os.chdir('../')
                else:
                    corners = tangent_image_corners(base_order, sample_order)
                    t_featurepoint_b = time.perf_counter()
                    pts1, desc1 = process_image_to_keypoints(path_o, corners, scale_factor, base_order, sample_order, opt, mode)
                    pts2, desc2 = process_image_to_keypoints(path_r, corners, scale_factor, base_order, sample_order, opt, mode)
                    t_featurepoint_a = time.perf_counter()
                    if descriptor == "Proposed1":
                        pts1, desc1 = adjust_vertical_intensity(pts1, desc1, (512, 1024, 3))
                        pts2, desc2 = adjust_vertical_intensity(pts2, desc2, (512, 1024, 3))
                        pts1, pts2, desc1, desc2, scores1, scores2 = sort_key(pts1, pts2, desc1, desc2, int(pts1.shape[0]*0.7))
                    elif descriptor == "Proposed2":
                        pts1, desc1 = adjust_vertical_intensity(pts1, desc1, (512, 1024, 3))
                        pts2, desc2 = adjust_vertical_intensity(pts2, desc2, (512, 1024, 3))
                        pts1, pts2, desc1, desc2, scores1, scores2 = sort_key(pts1, pts2, desc1, desc2, int(pts1.shape[0]*0.8))
                    elif descriptor == "Proposed3":
                        pts1, desc1 = adjust_vertical_intensity(pts1, desc1, (512, 1024, 3))
                        pts2, desc2 = adjust_vertical_intensity(pts2, desc2, (512, 1024, 3))
                        pts1, pts2, desc1, desc2, scores1, scores2 = sort_key(pts1, pts2, desc1, desc2, int(pts1.shape[0]*0.9))   
                    elif descriptor == "Proposed4":
                        pts1, desc1 = adjust_vertical_intensity(pts1, desc1, (512, 1024, 3))
                        pts2, desc2 = adjust_vertical_intensity(pts2, desc2, (512, 1024, 3))
                        pts1, pts2, desc1, desc2, scores1, scores2 = sort_key(pts1, pts2, desc1, desc2, int(pts1.shape[0]*0.95))                
                    else:
                        pts1, pts2, desc1, desc2, score1, score2 = sort_key(pts1, pts2, desc1, desc2, args.points)

                    
                
                if len(pts1.shape) == 1:
                    pts1 = pts1.reshape(1,-1)
                if len(pts2.shape) == 1:
                    pts2 = pts2.reshape(1,-1)

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
        base_path = f'results/data_real/FP_{args.points}/values/{scene}/{args.pose}/'+descriptor+'_'+args.inliers+'_'+args.solver
        os.system('mkdir -p '+base_path)
        np.savetxt(base_path+'/R_ERRORS.csv',np.array(R_ERROR[indicador]),delimiter=",")
        np.savetxt(base_path+'/T_ERRORS.csv',np.array(T_ERROR[indicador]),delimiter=",")
        np.savetxt(base_path+'/TIMES_FP.csv',np.array(TIMES_FP[indicador]),delimiter=",")
        np.savetxt(base_path+'/TIMES_MC.csv',np.array(TIMES_MC[indicador]),delimiter=",")
        np.savetxt(base_path+'/TIMES_PE.csv',np.array(TIMES_PE[indicador]),delimiter=",")
        np.savetxt(base_path+'/MATCHING_SCORE.csv',np.array(MATCHING_SCORE[indicador]),delimiter=",")
        np.savetxt(base_path+'/MEAN_MATCHING_ACCURCY.csv',np.array(MEAN_MATCHING_ACCURCY[indicador]),delimiter=",")
        np.savetxt(base_path+'/MATCHING_NUM.csv',np.array(MATCHING_NUM[indicador]),delimiter=",")
        np.savetxt(base_path+'/FP_NUM.csv',np.array(FP_NUM[indicador]),delimiter=",")
print('finish')

def adjust_vertical_intensity(pts1_, desc1_, img_shape):
    img_height = img_shape[0]
    center_y = img_height / 2.0

    for i, pt in enumerate(pts1_):
        distance_from_center_y = abs(pt[1] - center_y)
        adjustment_factor = np.sqrt(1 - (distance_from_center_y / center_y)**2)
        if distance_from_center_y == center_y:
            adjustment_factor = 1 - 1

        pts1_[i, 2] *= adjustment_factor
    
    return pts1_, desc1_


if __name__ == '__main__':
    main()