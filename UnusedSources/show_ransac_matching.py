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

sys.path.append(os.getcwd()+'/SPHORB-master')

import build1.sphorb_cpp as sphorb
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():

    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--points', type=int, default = 1000)
    parser.add_argument('--match', default="MNN")
    parser.add_argument('--g_metrics',default="False")
    parser.add_argument('--solver', default="GSM_wRT")
    parser.add_argument('--inliers', default="8PA")
    parser.add_argument('--descriptor', default = 'sift')
    parser.add_argument('--path', default = "./data/data_100/Room/97")
    args = parser.parse_args()


    print('X')

    path = args.path
    descriptor = args.descriptor
    img_hw = (512, 1024)
    path_o = path + f'/O.png'
    path_r = path + f'/R.png'

    opt, mode, sphered = get_descriptor(descriptor)

    method_idx = 0.0
    base_order = 0  # Base sphere resolution
    sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
    scale_factor = 1.0  # How much to scale input equirectangular image by
    save_ply = False  # Whether to save the PLY visualizations too
    dim = np.array([2*sphered, sphered])
    
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
        #corners = tangent_image_corners(base_order, sample_order)
        pts1_, desc1_, make_map_time, remap_time, feature_time = process_image_to_keypoints(path_o, scale_factor, base_order, sample_order, opt, mode, img_hw)
        pts2_, desc2_, make_map_time, remap_time, feature_time = process_image_to_keypoints(path_r, scale_factor, base_order, sample_order, opt, mode, img_hw)
        

    else:           
        os.chdir('SPHORB-master/')
        path_o = "."+path_o
        path_r = "."+path_r
        pts1, desc1 = get_kd(sphorb.sphorb(path_o, args.points))
        pts2, desc2 = get_kd(sphorb.sphorb(path_r, args.points))
        pts1, desc1 = convert_sphorb(pts1, desc1)
        pts2, desc2 = convert_sphorb(pts2, desc2)
        os.chdir('../')

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
    s_pts1, s_pts2, x1_, x2_ = matched_points(pts1, pts2, desc1, desc2, "100p", opt, match=args.match, constant=method_idx)
    print(s_pts1.shape, x1_.shape)

    x1,x2 = coord_3d(x1_, dim), coord_3d(x2_, dim)
    s_pts1, s_pts2 = coord_3d(s_pts1, dim), coord_3d(s_pts2, dim)

    E, cam, inlier_idx = get_cam_pose_by_ransac(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers, solver="SK")
    R1_,R2_,T1_,T2_ = decomposeE(E.T)
    R_,T_ = choose_rt(R1_,R2_,T1_,T2_,x1,x2)
    print("True:", sum(inlier_idx), len(inlier_idx), ", ratio:", sum(inlier_idx) / len(inlier_idx))

    
    vis_img = plot_matches(img_o, img_r, s_pts1[:, :2], s_pts2[:, :2], x1_[:, :2], x2_[:, :2], inlier_idx)
    vis_img = cv2.resize(vis_img,dsize=(512,512))
    cv2.imshow("aaa", vis_img)
    c = cv2.waitKey()



def convert_sphorb(pts, desc):
    pts_new = np.hstack((pts, pts[:, 2:3]))
    pts_tensor = torch.tensor(pts_new)
    desc_tensor = torch.tensor(desc.T)
    
    return pts_tensor, desc_tensor


def transform_and_evaluate(x1, x2, R, t, threshold):
    points1 = x1[:, :3]
    points2 = x2[:, :3]

    # 座標をホモジニアス座標に変換
    points1_hom = np.hstack([points1, np.ones((points1.shape[0], 1))])
    # 変換行列を作成（回転と並進を合わせた行列）
    transform_matrix = np.hstack([R, t.reshape(-1, 1)])

    # 点群x1を変換
    transformed_points1_hom = points1_hom @ transform_matrix.T

    # 3D座標から2Dプロジェクション（ここでは単純化のためにz座標を無視）
    transformed_points1 = transformed_points1_hom[:, :2]
    points2_2d = points2[:, :2]
    print(transformed_points1[0:3], points2_2d[0:3])
    # 各点の誤差を計算
    errors = np.sqrt(np.sum((transformed_points1 - points2_2d) ** 2, axis=1))
    mse = np.mean(errors ** 2)

    # 閾値以下の誤差を持つ点の数と割合
    under_threshold = errors < threshold
    count_under_threshold = np.sum(under_threshold)
    ratio_under_threshold = count_under_threshold / len(errors)

    return mse, count_under_threshold, ratio_under_threshold




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
        thickness = 1
    else:
        thickness = 1
    
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




if __name__ == '__main__':
    main()