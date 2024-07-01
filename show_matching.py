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
    Rx = np.load(path+"/R.npy")
    Tx = np.load(path+"/T.npy")
    Tx_norm = Tx / np.linalg.norm(Tx)

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

    img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    

    if opt != 'sphorb':
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
    pts1, desc1, _ = sort_key_div(pts1_, desc1_, num_points_1)   
    pts2, desc2, _ = sort_key_div(pts2_, desc2_, num_points_2)  


    if len(pts1.shape) == 1:
        pts1 = pts1.reshape(1,-1)
    s_pts1, s_pts2, x1_, x2_ = matched_points(pts1, pts2, desc1, desc2, "100p", opt, match=args.match, constant=method_idx)
    x1,x2 = coord_3d(x1_, dim), coord_3d(x2_, dim)
    s_pts1, s_pts2 = coord_3d(s_pts1, dim), coord_3d(s_pts2, dim)
    E, can, inlier_idx = get_cam_pose_by_ransac_GSM_const_wRT(x1.copy().T,x2.copy().T, get_E = True, I = args.inliers)
    print(inlier_idx)
    R1_,R2_,T1_,T2_ = decomposeE(E.T)
    R_,T_ = choose_rt(R1_,R2_,T1_,T2_,x1,x2)
    E_true = compute_essential_matrix(Rx, Tx_norm)
    results = evaluate_matches(x1, x2, E_true)

    print(Tx)
    print(x1.shape, x2.shape)
    #print(E_true)
    #print(Rx)
    #print(Tx_norm)
    #x1_world = np.dot(Rx, (x2-Tx).T).T
    #x1_world_norm = x1_world / np.linalg.norm(x1_world, axis=1).reshape(-1, 1)
    #print(x1_world[0:3])
    #x_a = np.dot(Rx, x1.T).T+Tx.T
    #x_a = x_a / np.linalg.norm(x_a, axis=1).reshape(-1, 1)
    #x_b = np.dot(Rx, x1.T).T-Tx.T
    #x_b = x_b / np.linalg.norm(x_b, axis=1).reshape(-1, 1)
    #x_c = np.dot(Rx, x2.T).T+Tx.T
    #x_c = x_c / np.linalg.norm(x_c, axis=1).reshape(-1, 1)
    #x_d = np.dot(Rx, x2.T).T-Tx.T
    #x_d = x_d / np.linalg.norm(x_d, axis=1).reshape(-1, 1)
    #print(x2_world[0:3])
    #print(x1[0])
    #print(x2[0:2])
    #print()
    #print(x_a[0:2])
    #print(x_b[0])
    #print(x_c[0])
    #print(x_d[0])

    print("Evaluation Results:")
    print(f"Valid Matches: {results['valid_matches']} / {results['total_matches']}")
    print(f"Valid Ratio: {results['valid_ratio']*100:.2f}%")
    print(f"MSE: {results['mse']}")

    plot_epipolar_results(results['epipolar_results'])
    vis_img = plot_matches(img_o, img_r, s_pts1[:, :2], s_pts2[:, :2], x1_[:, :2].copy(), x2_[:, :2].copy(), results['threshold_results'])
    vis_img = cv2.resize(vis_img,dsize=(512,512))
    cv2.imshow("aaa", vis_img)
    c = cv2.waitKey()
    for i in range(len(x1)):
        vis_img = plot_match(img_o, img_r, s_pts1[i, :2], s_pts2[i, :2], x1_[i, :2].copy(), x2_[i, :2].copy(), results['threshold_results'][i])
        vis_img = cv2.resize(vis_img,dsize=(512,512))
        print(x1[i], x2[i])
        cv2.imshow("aaa", vis_img)
        c = cv2.waitKey()
        if c == 27: # esc
            break


def evaluate_matches(x1, x2, E, threshold=1):
    epipolar_results = np.einsum('ij,jk,ik->i', x2, E, x1)*180/np.pi
    #print(epipolar_results)
    #angles = np.arccos(np.clip(epipolar_results, -1, 1))
    #max_epipolar_result = np.max(angles)
    #min_epipolar_result = np.min(angles)
    #print(max_epipolar_result, min_epipolar_result)
    valid_matches = np.sum(abs(epipolar_results) < threshold)
    total_matches = len(epipolar_results)
    valid_ratio = valid_matches / total_matches
    mse = np.mean(epipolar_results**2)

    threshold_results = np.where(abs(epipolar_results) < threshold, 1, 0)
    
    return {
        "valid_matches": valid_matches,
        "total_matches": total_matches,
        "valid_ratio": valid_ratio,
        "mse": mse,
        "threshold_results": threshold_results,
        "epipolar_results": epipolar_results
    }


def plot_epipolar_results(epipolar_results):
    plt.figure()
    range = (-0.05*180/np.pi, 0.05*180/np.pi)
    plt.hist(epipolar_results, bins=50, color='blue', alpha=0.7, range=range)
    plt.title("Epipolar Results Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.xlim(range) 
    plt.grid(True)
    plt.savefig("temp_histogram.png")
    plt.close()
    
    vis_img = cv2.imread("temp_histogram.png")
    cv2.imshow("Epipolar Results Histogram", vis_img)
    c = cv2.waitKey()
    if c == 27:  # Escキーで閉じる
        cv2.destroyAllWindows()


def cross_product_matrix(t):
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])

def compute_essential_matrix(R, t):
    t_cross = cross_product_matrix(t)
    E = t_cross.dot(R)
    return E


def plot_match(image0,
                 image1,
                 kpt0,
                 kpt1,
                 x1,
                 x2,
                 match_flag,
                 radius=2):
    
    out0 = plot_keypoints(image0, kpt0, radius, (255, 0, 0))
    out1 = plot_keypoints(image1, kpt1, radius, (255, 0, 0))

    H0, W0 = image0.shape[0], image0.shape[1]
    H1, W1 = image1.shape[0], image1.shape[1]

    H, W = H0 + H1, max(W0, W1)
    out = 255 * np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = out0
    out[H0:H0+H1, :W1, :] = out1

    mkpts0, mkpts1 = x1, x2
    mkpts0 = np.round(mkpts0).astype(int)
    mkpts1 = np.round(mkpts1).astype(int)
    mkpts1[1] += H0

    if image0.shape[0] > 2000:
        thickness = 10
    if image0.shape[0] > 1000:
        thickness = 2
    else:
        thickness = 2
    (xs, ys), (xe, ye) = mkpts0, mkpts1
    if match_flag:
        mcolor = (0, 255, 0)
    else:
        mcolor = (0, 0, 255)
    cv2.line(out, (xs%W0, ys), (xe%W1, ye),
             color=mcolor,
             thickness=thickness,
            )

    return out


def plot_matches(image0,
                 image1,
                 kpts0,
                 kpts1,
                 x1,
                 x2,
                 inlier_idx,
                 radius=2):
    
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

    return out


if __name__ == '__main__':
    main()