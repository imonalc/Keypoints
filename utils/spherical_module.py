import cv2

import pandas as pd
import numpy as np


from utils.ransac   import *
from utils.keypoint import *
from utils.metrics  import *
from utils.camera_recovering import *
from scipy.spatial import distance

import time

import sys


def make_image_map(img_hw, rad=np.pi/2):
    (h, w) = img_hw
    w_half = int(w / 2)
    h_half = int(h / 2)

    phi, theta = np.meshgrid(np.linspace(-np.pi, np.pi, w_half*2),
                             np.linspace(-np.pi/2, np.pi/2, h_half*2))

    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)

    xx, yy, zz = rotate_yaw(x, y, z, rad)
    xx, yy, zz = rotate_pitch(xx, yy, zz, rad)
    xx, yy, zz = rotate_yaw(xx, yy, zz, rad)

    theta = np.arcsin(zz)
    phi = np.arctan2(yy, xx)

    Y_remap = 2 * theta / np.pi * h_half + h_half
    X_remap = phi / np.pi * w_half + w_half

    return Y_remap, X_remap



def remap_image(image_path, output_path, YX_remap):
    (Y_remap, X_remap) = YX_remap
    img = cv2.imread(image_path)
    t_b = time.perf_counter()
    out = cv2.remap(img, X_remap.astype(np.float32), Y_remap.astype(np.float32), 
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    t_a = time.perf_counter()
    ret = t_a - t_b
    cv2.imwrite(output_path, out)

    return ret




def round_coordinates(tensor):
    coords_int = tensor[:, :2].int()
    other_data = tensor[:, 2:]
    rounded_tensor = torch.cat((coords_int, other_data), dim=1)
    return rounded_tensor


def filter_middle_latitude(pts_, desc_, img_hw, invert_mask=False):
    spherical_coords = equirectangular_to_spherical_coords(img_hw)
    x_indices = pts_[:, 0].long()
    y_indices = pts_[:, 1].long()

    theta_values = spherical_coords[y_indices, x_indices, 0]
    mask = (torch.pi/4 <= theta_values) & (theta_values < 3*torch.pi/4)
    if invert_mask:
        mask = ~mask

    pts = pts_[mask]
    desc = desc_.T[mask].T

    return pts, desc


def equirectangular_to_spherical_coords(img_hw, device='cpu'):
    img_height = img_hw[0]
    img_width = img_hw[1]
    theta = torch.linspace(0, np.pi, img_height, device=device)
    phi = torch.linspace(0, 2 * np.pi, img_width, device=device) 
    phi_grid, theta_grid = torch.meshgrid(phi, theta, indexing="xy")
    return torch.stack([theta_grid, phi_grid, torch.ones_like(theta_grid)], dim=-1)


def spherical_to_cartesian(phi, theta):
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return x, y, z

def rotate_coordinates(x, y, z, angle=np.pi/2):
    xx = x * np.cos(angle) + z * np.sin(angle)
    yy = y
    zz = -x * np.sin(angle) + z * np.cos(angle)
    return xx, yy, zz

def cartesian_to_spherical(x, y, z):
    theta = np.arcsin(z)
    phi = np.arctan2(y, x)
    return phi, theta


def convert_coordinate(input_xy, image_size_hw):
    h, w = image_size_hw
    w_half = w / 2
    h_half = h / 2
    
    phi = (input_xy[0] - w_half) * np.pi * 2 /w
    theta = (input_xy[1] - h_half) * np.pi /h
    
    x, y, z = spherical_to_cartesian(phi, theta)
    xx, yy, zz = rotate_coordinates(x, y, z)
    new_phi, new_theta = cartesian_to_spherical(xx, yy, zz)

    new_y = 2*new_theta * h_half / np.pi + h_half
    new_x = new_phi * w_half / np.pi + w_half
    
    return new_x, new_y


def convert_coordinates_vectorized(tensor, image_size_hw):
    h, w = image_size_hw
    w_half = w / 2
    h_half = h / 2

    x_coords = tensor[:, 0]
    y_coords = tensor[:, 1]

    phi = (x_coords - w_half) * np.pi * 2 / w
    theta = (y_coords - h_half) * np.pi / h
    
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)

    angle = np.pi / 2
    xx = x * np.cos(angle) + z * np.sin(angle)
    yy = y
    zz = -x * np.sin(angle) + z * np.cos(angle)

    new_theta = np.arcsin(zz)
    new_phi = np.arctan2(yy, xx)

    new_y = new_theta * h_half / (np.pi/2) + h_half
    new_x = new_phi * w_half / np.pi + w_half
    
    transformed_tensor = tensor
    transformed_tensor[:, 0] = new_x
    transformed_tensor[:, 1] = new_y
    
    return transformed_tensor



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


def make_image_map(img_hw, rad=np.pi/2):
    (h, w) = img_hw
    w_half = int(w / 2)
    h_half = int(h / 2)

    phi, theta = np.meshgrid(np.linspace(-np.pi, np.pi, w_half*2),
                             np.linspace(-np.pi/2, np.pi/2, h_half*2))

    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)

    xx, yy, zz = rotate_yaw(x, y, z, rad)
    xx, yy, zz = rotate_pitch(xx, yy, zz, rad)
    xx, yy, zz = rotate_yaw(xx, yy, zz, rad)

    theta = np.arcsin(zz)
    phi = np.arctan2(yy, xx)

    Y_remap = 2 * theta / np.pi * h_half + h_half
    X_remap = phi / np.pi * w_half + w_half

    return Y_remap, X_remap
