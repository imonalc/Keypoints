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
    parser.add_argument('--points', type=int, default = 12000)
    parser.add_argument('--match', default="ratio")
    parser.add_argument('--solver', default="None")
    parser.add_argument('--inliers', default="8PA")
    parser.add_argument('--datas'      , nargs='+')
    parser.add_argument('--descriptors', nargs='+')
    args = parser.parse_args()


    DATAS       = args.datas
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
    data_name = "_BC2"
    np.random.seed(0)
    data = get_data(DATAS)
    for data in DATAS:

        mypath = os.path.join('data/data_100',data)
        paths  = [os.path.join(os.getcwd(),'data/data_100',data,f) for f in listdir('data/data_100/'+data) if isdir(join(mypath, f))]
        NUM = NUM + len(paths)
        std = []

        for path in tqdm(paths):
            make_BCG = not os.path.isfile(path + f'/R{data_name}.png')
            if make_BCG:
                path_o = path + '/O.png'
                base_order = 0  # Base sphere resolution
                sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
                scale_factor = 1.0  # How much to scale input equirectangular image by
                img_o = load_torch_img(path_o)[:3, ...].float()
                img_o = F.interpolate(img_o.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
                img_o = torch2numpy(img_o.byte())
                img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
                alpha = np.random.uniform(1/2, 2.0)
                beta = np.random.randint(-50, 50)
                img_o = cv2.convertScaleAbs(img_o, alpha=alpha, beta=beta)
                img_o_pil = Image.fromarray(img_o)
                img_o_pil.save(path + f"/O{data_name}.png")
                img_o = load_torch_img(path_o)[:3, ...].float()
                img_o = F.interpolate(img_o.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
                img_o = torch2numpy(img_o.byte())
                img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)

                path_r = path + '/R.png'
                base_order = 0  # Base sphere resolution
                sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
                scale_factor = 1.0  # How much to scale input equirectangular image by
                img_r = load_torch_img(path_r)[:3, ...].float()
                img_r = F.interpolate(img_r.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
                img_r = torch2numpy(img_r.byte())
                img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
                alpha = np.random.uniform(0.5, 2.0)
                beta = np.random.randint(-50, 50)
                img_r = cv2.convertScaleAbs(img_r, alpha=alpha, beta=beta)
                img_r_pil = Image.fromarray(img_r)
                img_r_pil.save(path + f"/R{data_name}.png")
                img_r = load_torch_img(path_r)[:3, ...].float()
                img_r = F.interpolate(img_r.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
                img_r = torch2numpy(img_r.byte())
                img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)

            for indicador, descriptor in enumerate(DESCRIPTORS):



                try:
                    opt, mode, sphered, method_idx = get_descriptor(descriptor)

                    base_order = 0  # Base sphere resolution
                    sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
                    scale_factor = 1.0  # How much to scale input equirectangular image by
                    save_ply = False  # Whether to save the PLY visualizations too
                    dim = np.array([2*sphered, sphered])

                    path_o = path + f'/O{data_name}.png'
                    path_r = path + f'/R{data_name}.png'

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

                        pts1, pts2, desc1, desc2, score1, score2 = sort_key(pts1, pts2, desc1, desc2, args.points)


                    
                    

                    if len(pts1.shape) == 1:
                        pts1 = pts1.reshape(1,-1)

                    if len(pts2.shape) == 1:
                        pts2 = pts2.reshape(1,-1)
                    Rx = np.load(path+"/R.npy")
                    Tx = np.load(path+"/T.npy")


                    len_pts = (len(pts1) + len(pts2)) / 2


                    if pts1.shape[0] > 0 or pts2.shape[0] >0:
                        if method_idx == 100:
                            img_o = load_torch_img(path_o)[:3, ...].float()
                            img_o = F.interpolate(img_o.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
                            img_r = load_torch_img(path_r)[:3, ...].float()
                            img_r = F.interpolate(img_r.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
                            img_o = torch2numpy(img_o.byte())
                            img_r = torch2numpy(img_r.byte())

                            img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
                            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
                            s_pts1, s_pts2, x1_, x2, t_matching_b, t_matching_a = superglue_matching(pts1, pts2, desc1, desc2, score1, score2, img_o, img_r, device)
                        elif method_idx == 101:
                            s_pts1, s_pts2, x1_, x2, t_matching_b, t_matching_a = lightglue_matching(pts1, pts2, desc1, desc2, score1, score2, device)
                        else:
                            t_matching_b = time.perf_counter()
                            s_pts1, s_pts2, x1_, x2_ = matched_points(pts1, pts2, desc1, desc2, "100p", opt, args.match, use_new_method=method_idx)
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
            base_path = f'results/FP_{args.points}{data_name}/values/'+data+'_'+descriptor+'_'+args.inliers+'_'+args.solver
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




if __name__ == '__main__':
    main()




