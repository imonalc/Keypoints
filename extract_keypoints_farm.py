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
    parser.add_argument('--points', type=int, default = 500)
    parser.add_argument('--match', default="ratio")
    parser.add_argument('--g_metrics',default="False")
    parser.add_argument('--solver', default="GSM_wRT")
    parser.add_argument('--inliers', default="8PA")
    parser.add_argument('--pose'      , default="pose_test")
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
    TIMES = []
    st = time.time()
    path_true_value = os.path.join("./data/Farm/new/Calibration/", pose)
    RQ_true_value = np.genfromtxt(path_true_value + "/RQ_true_value.csv", delimiter=',')
    T_true_value = np.genfromtxt(path_true_value + "/T_true_value.csv", delimiter=',')
    Rx = np.genfromtxt(path_true_value + "/Rm_true_value.csv", delimiter=',')
    Tx = np.genfromtxt(path_true_value + "/T_true_value.csv", delimiter=',')

    mypath = os.path.join('./data/Farm/new',pose)
    paths  = [os.path.join('./data/Farm/new/',pose,f) for f in listdir('./data/Farm/new/'+pose) if isdir(join(mypath, f))]
    print(os.path.join('./data/Farm/new',pose))
    NUM = NUM + len(paths)
    print(paths)

    std = []
    for path in tqdm(paths):
        #print(path)

        for indicador, descriptor in enumerate(DESCRIPTORS):


            try:
                opt, mode, sphered, method_idx = get_descriptor(descriptor)
                base_order = 0  # Base sphere resolution
                sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
                scale_factor = 1.0  # How much to scale input equirectangular image by
                save_ply = False  # Whether to save the PLY visualizations too
                dim = np.array([2*sphered, sphered])
                path_o = path + '/O.png'
                path_r = path + '/R.png'
                if opt != 'sphorb':
                    # ----------------------------------------------
                    # Compute necessary data
                    # ----------------------------------------------
                    # 80 baricenter points
                    corners = tangent_image_corners(base_order, sample_order)
                    t_featurepoint_b = time.perf_counter()
                    pts1, desc1 = process_image_to_keypoints(path_o, corners, scale_factor, base_order, sample_order, opt, mode)
                    pts2, desc2 = process_image_to_keypoints(path_r, corners, scale_factor, base_order, sample_order, opt, mode)
                    t_featurepoint_a = time.perf_counter()
                    pts1, pts2, desc1, desc2, _, _ = sort_key(pts1, pts2, desc1, desc2, args.points)
                    
                else:
                    os.chdir('SPHORB-master/')
                    t_featurepoint_b = time.perf_counter()
                    path_o = "."+path_o
                    path_r = "."+path_r
                    pts1, desc1 = get_kd(sphorb.sphorb(path_o, args.points))
                    pts2, desc2 = get_kd(sphorb.sphorb(path_r, args.points))
                    t_featurepoint_a = time.perf_counter()
                    os.chdir('../')
                
                len_pts = (len(pts1) + len(pts2)) / 2

                #print(pts1.shape, desc1.shape)

                height_threshold = 512*0.9
                cond1_1 = (pts1[:, 1] < height_threshold)
                cond2_1 = (pts2[:, 1] < height_threshold)
                if pose == "pose1":
                    cond1_2 = ~(((400 < pts1[:, 0]) &(pts1[:, 0] < 840))  & (pts1[:, 1] > 400))
                    cond1_3 = ~(((680 < pts1[:, 0]) &(pts1[:, 0] < 800)) & ((220< pts1[:, 1])))
                    cond2_2 = ~((pts2[:, 0] < 400) & (pts2[:, 1] > 350))
                    cond2_3 = ~(((220 < pts2[:, 0]) &(pts2[:, 0] < 320)) & ((250< pts2[:, 1])))
                elif pose == "pose2":
                    cond1_2 = ~(((500 < pts1[:, 0]) &(pts1[:, 0] < 1000))  & (pts1[:, 1] > 360))
                    cond1_3 = ~(((800 < pts1[:, 0]) &(pts1[:, 0] < 1000)) & ((220< pts1[:, 1])))
                    cond2_2 = ~((pts2[:, 0] < 400) & (pts2[:, 1] > 350))
                    cond2_3 = ~(((100 < pts2[:, 0]) &(pts2[:, 0] < 250)) & ((250< pts2[:, 1])))
                elif pose == "pose3":
                    cond1_2 = ~(((200 < pts1[:, 0]) &(pts1[:, 0] < 700))  & (pts1[:, 1] > 360))
                    cond1_3 = ~(((500 < pts1[:, 0]) &(pts1[:, 0] < 750)) & ((240< pts1[:, 1])))
                    cond2_2 = ~((pts2[:, 0] < 400) & (pts2[:, 1] > 300))
                    cond2_3 = ~(((200 < pts2[:, 0]) &(pts2[:, 0] < 400)) & ((250< pts2[:, 1])))
                elif pose == "pose4":
                    cond1_2 = ~(((500 < pts1[:, 0]) &(pts1[:, 0] < 1000))  & (pts1[:, 1] > 360))
                    cond1_3 = ~(((800 < pts1[:, 0]) &(pts1[:, 0] < 1000)) & ((240< pts1[:, 1])))
                    cond2_2 = ~((pts2[:, 0] < 300) & (pts2[:, 1] > 300))
                    cond2_3 = ~(((60 < pts2[:, 0]) &(pts2[:, 0] < 240)) & ((250< pts2[:, 1])))
                elif pose == "pose5":
                    cond1_2 = ~(((400 < pts1[:, 0]) &(pts1[:, 0] < 900))  & (pts1[:, 1] > 360))
                    cond1_3 = ~(((650 < pts1[:, 0]) &(pts1[:, 0] < 850)) & ((240< pts1[:, 1])))
                    cond2_2 = ~((pts2[:, 0] < 400) & (pts2[:, 1] > 400))
                    cond2_3 = ~(((100 < pts2[:, 0]) &(pts2[:, 0] < 350)) & ((250< pts2[:, 1])))
    
                valid_idx1 = cond1_1 & cond1_2 &cond1_3
                pts1 =  pts1[valid_idx1]
                desc1 = desc1[valid_idx1]
                valid_idx2 = cond2_1 & cond2_2 &cond2_3
                pts2 =  pts2[valid_idx2]
                desc2 = desc2[valid_idx2]


                if len(pts1.shape) == 1:
                    pts1 = pts1.reshape(1,-1)
                if len(pts2.shape) == 1:
                    pts2 = pts2.reshape(1,-1)


                if pts1.shape[0] > 0 or pts2.shape[0] >0:
                    t_matching_b = time.perf_counter()
                    s_pts1, s_pts2, x1, x2 = matched_points(pts1, pts2, desc1, desc2, "100p", opt, args.match, use_new_method=method_idx)
                    t_matching_a = time.perf_counter()
                    x1,x2 = coord_3d(x1, dim), coord_3d(x2, dim)
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
                        #print("True:", sum(inlier_idx), len(inlier_idx), ", ratio:", sum(inlier_idx) / len(inlier_idx))
                        R1_,R2_,T1_,T2_ = decomposeE(E.T)
                        R_,T_ = choose_rt(R1_,R2_,T1_,T2_,x1,x2)
                        

                        #print(R_)
                        #print(T_)

                    
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
        os.system('mkdir -p '+f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver)
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/R_ERRORS.csv',np.array(R_ERROR[indicador]),delimiter=",")
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/T_ERRORS.csv',np.array(T_ERROR[indicador]),delimiter=",")
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/TIMES_FP.csv',np.array(TIMES_FP[indicador]),delimiter=",")
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/TIMES_MC.csv',np.array(TIMES_MC[indicador]),delimiter=",")
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/TIMES_PE.csv',np.array(TIMES_PE[indicador]),delimiter=",")
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/MATCHING_SCORE.csv',np.array(MATCHING_SCORE[indicador]),delimiter=",")
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/MEAN_MATCHING_ACCURCY.csv',np.array(MEAN_MATCHING_ACCURCY[indicador]),delimiter=",")
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/MATCHING_NUM.csv',np.array(MATCHING_NUM[indicador]),delimiter=",")
        np.savetxt(f'results/FP_{args.points}/values/'+args.pose+'_'+descriptor+'_'+args.inliers+'_'+args.solver+'/FP_NUM.csv',np.array(FP_NUM[indicador]),delimiter=",")
print('finish')




def mat2quat(mat):
    rot = Rotation.from_matrix(mat)
    quat = rot.as_quat()
    return quat


def mean_quat(R_quat_list):
    x = np.array([R_quat for R_quat in R_quat_list])
    m = x.T @ x
    w, v = np.linalg.eig(m)
    return v[:, np.argmax(w)]



if __name__ == '__main__':
    main()