import copy
import os
import cv2
import glob
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch

from utils.ransac   import *
from utils.keypoint import *
from utils.metrics  import *
from utils.method  import *
from utils.camera_recovering import *
from utils.matching import *
from utils.spherical_module import *
import time


def make_image_map(img_hw, rad=torch.pi/2):
    make_map_time1 = time.perf_counter()
    (h, w) = img_hw
    w_half = int(w / 2)
    h_half = int(h / 2)

    phi, theta = np.meshgrid(np.linspace(-torch.pi, torch.pi, w_half*2),
                             np.linspace(-torch.pi/2, torch.pi/2, h_half*2))

    x, y, z = spherical_to_cartesian(phi, theta)

    xx, yy, zz = rotate_yaw(x, y, z, rad)
    xx, yy, zz = rotate_pitch(xx, yy, zz, rad)
    xx, yy, zz = rotate_yaw(xx, yy, zz, rad)

    new_phi, new_theta = cartesian_to_spherical(xx, yy, zz)

    Y_remap = 2 * new_theta / torch.pi * h_half + h_half
    X_remap = new_phi / torch.pi * w_half + w_half

    make_map_time2 = time.perf_counter()
    make_map_time = make_map_time2 - make_map_time1

    return Y_remap, X_remap, make_map_time


def rotate_yaw(x, y, z, rot):
    xx = x * np.cos(rot) - y * np.sin(rot)
    yy = x * np.sin(rot) + y * np.cos(rot)
    zz = z

    return xx, yy, zz

def rotate_roll(x, y, z, rot):
    xx = x
    yy = y * np.cos(rot) - z * np.sin(rot)
    zz = y * np.sin(rot) + z * np.cos(rot)

    return xx, yy, zz

def rotate_pitch(x, y, z, rot):
    xx = x * np.cos(rot) + z * np.sin(rot)
    yy = y
    zz = -x * np.sin(rot) + z * np.cos(rot)

    return xx, yy, zz


def spherical_to_cartesian(phi, theta):
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    theta = np.arcsin(z)
    phi = np.arctan2(y, x)
    return phi, theta


def remap_crop_image(img, YX_remap, img_hw_crop, crop_start_xy):
    (Y_remap, X_remap) = YX_remap

    remap_time1 = time.perf_counter()
    img2 = cv2.remap(img, X_remap.astype(np.float32), Y_remap.astype(np.float32), 
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    remap_time2 = time.perf_counter()
    remap_time = remap_time2 - remap_time1

    img1_cropped = img[crop_start_xy[0]:crop_start_xy[0]+img_hw_crop[0], crop_start_xy[1]:crop_start_xy[1]+img_hw_crop[1]]
    img2_cropped = img2[crop_start_xy[0]:crop_start_xy[0]+img_hw_crop[0], crop_start_xy[1]:crop_start_xy[1]+img_hw_crop[1]]


    return img1_cropped, img2_cropped, remap_time


def coord_3d(X,dim):
    phi   = X[:,1]/dim[1] * np.pi     # phi
    theta = X[:,0]/dim[0] * 2 * np.pi         # theta
    R = np.stack([(np.sin(phi) * np.cos(theta)).T,(np.sin(phi) * np.sin(theta)).T,np.cos(phi).T], axis=1)

    return R