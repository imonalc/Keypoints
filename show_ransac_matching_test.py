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
    parser.add_argument('--match', default="BF_KNN")
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
    path_o2 = path + f'/O2.png'
    path_r2 = path + f'/R2.png'
    path_op = path + f'/Op.png'
    path_rp = path + f'/Rp.png'
    path_op2 = path + f'/Op2.png'
    path_rp2 = path + f'/Rp2.png'


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

    method_idx = 0.0
    base_order = 0  # Base sphere resolution
    sample_order = 8  # Determines sample resolution (10 = 2048 x 4096)
    scale_factor = 1.0  # How much to scale input equirectangular image by
    save_ply = False  # Whether to save the PLY visualizations too
    dim = np.array([2*sphered, sphered])

    Y_remap, X_remap = make_image_map(img_hw)
    remap_t1 = remap_image(path_o, path_o2, (Y_remap, X_remap))
    remap_t2 = remap_image(path_r, path_r2, (Y_remap, X_remap))
    
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
        

    else:           
        os.chdir('SPHORB-master/')
        path_o = "."+path_o
        path_r = "."+path_r
        t_featurepoint_b = time.perf_counter()
        pts1, desc1 = get_kd(sphorb.sphorb(path_o, args.points))
        pts2, desc2 = get_kd(sphorb.sphorb(path_r, args.points))
        pts1, desc1 = convert_sphorb(pts1, desc1)
        pts2, desc2 = convert_sphorb(pts2, desc2)
        t_featurepoint_a = time.perf_counter()
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

    t_matching_b = time.perf_counter()
    s_pts1, s_pts2, x1_, x2_ = matched_points(pts1, pts2, desc1, desc2, "100p", opt, match=args.match, constant=method_idx)
    t_matching_a = time.perf_counter()
    
    #print(x1_.shape, s_pts1.shape)
    #print(x1_)
    x1,x2 = coord_3d(x1_, dim), coord_3d(x2_, dim)
    s_pts1, s_pts2 = coord_3d(s_pts1, dim), coord_3d(s_pts2, dim)

    E, can, inlier_idx = get_cam_pose_by_ransac_GSM_const_wRT(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
    R1_,R2_,T1_,T2_ = decomposeE(E.T)
    R_,T_ = choose_rt(R1_,R2_,T1_,T2_,x1,x2)
    print("True:", sum(inlier_idx), len(inlier_idx), ", ratio:", sum(inlier_idx) / len(inlier_idx))
    print("FP:", "{:.4g}".format((t_featurepoint_a-t_featurepoint_b)/2))
    print("MC:", "{:.4g}".format((t_matching_a-t_matching_b)))

    
    vis_img = plot_matches(img_o, img_r, s_pts1[:, :2], s_pts2[:, :2], x1_[:, :2], x2_[:, :2], inlier_idx)
    vis_img = cv2.resize(vis_img,dsize=(512,512))
    cv2.imshow("aaa", vis_img)
    c = cv2.waitKey()



def convert_sphorb(pts, desc):
    pts_new = np.hstack((pts, pts[:, 2:3]))
    pts_tensor = torch.tensor(pts_new)
    desc_tensor = torch.tensor(desc.T)
    
    return pts_tensor, desc_tensor



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