import torch
import pandas as pd
import numpy as np
import cv2


from utils.keypoint import *
from utils.matching import *
from utils.spherical_module import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def proposed_image_mapping(path_o, path_r, path_o2, path_r2, path_op, path_rp, path_op2, path_rp2, img_hw, padding_length):
    img_hw_crop = (img_hw[0]//2+padding_length*2+2, img_hw[1]*3//4+padding_length*2+2)
    crop_start_xy = ((img_hw[0]-img_hw_crop[0])//2 - 1, (img_hw[1]-img_hw_crop[1])//2 - 1)
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

    return img_hw_crop, crop_start_xy


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