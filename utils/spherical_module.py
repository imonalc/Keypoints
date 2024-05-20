import cv2
import numpy as np
import time

from utils.ransac   import *
from utils.keypoint import *
from utils.metrics  import *
from utils.camera_recovering import *
from scipy.spatial import distance



def make_image_map(img_hw, rad=np.pi/2):
    (h, w) = img_hw
    w_half = int(w / 2)
    h_half = int(h / 2)

    phi, theta = np.meshgrid(np.linspace(-np.pi, np.pi, w_half*2),
                             np.linspace(-np.pi/2, np.pi/2, h_half*2))

    x, y, z = spherical_to_cartesian(phi, theta)

    xx, yy, zz = rotate_yaw(x, y, z, rad)
    xx, yy, zz = rotate_pitch(xx, yy, zz, rad)
    xx, yy, zz = rotate_yaw(xx, yy, zz, rad)

    new_phi, new_theta = cartesian_to_spherical(xx, yy, zz)

    Y_remap = 2 * new_theta / np.pi * h_half + h_half
    X_remap = new_phi / np.pi * w_half + w_half

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



def filter_middle_latitude(pts_, desc_, img_hw, invert_mask=False):
    img_height = img_hw[0]
    img_width = img_hw[1]

    theta_values = (pts_[:, 1] / img_height) * np.pi
    phi_values = (pts_[:, 0] / img_width) * 2 * np.pi  # x座標を使用して経度を計算
    theta_std = 6
    phi_std = 8
    mask_theta = (torch.pi/theta_std <= theta_values) & (theta_values < (theta_std-1)*torch.pi/theta_std)
    mask_phi = (np.pi/phi_std <= phi_values) & (phi_values < (phi_std*2-1)*np.pi/phi_std)
    mask = mask_theta & mask_phi
    if invert_mask:
        mask = ~mask

    pts = pts_[mask]
    desc = desc_.T[mask].T

    return pts, desc


def create_filter_map(img_hw):
    img_height, img_width = img_hw
    img_height_half = img_height // 2
    img_width_half = img_width // 2

    filter_map = [[False for _ in range(img_width)] for _ in range(img_height)]
    for i in range(img_height):
        for j in range(img_width):
            theta = (i - img_height_half) / img_height * np.pi
            phi = (j - img_width_half) / img_width * 2 * np.pi
            x, y, z = spherical_to_cartesian(phi, theta)
            rad = np.pi / 2
            xx, yy, zz = rotate_yaw(x, y, z, rad)
            xx, yy, zz = rotate_pitch(xx, yy, zz, rad)
            xx, yy, zz = rotate_yaw(xx, yy, zz, rad)
            new_phi, new_theta = cartesian_to_spherical(xx, yy, zz)
            i2 = int((new_theta / (np.pi / 2) + 1) * img_height_half)
            j2 = int((new_phi / np.pi + 1) * img_width_half)
            if abs(j2 - img_width_half) > img_width * 3 // 8:
                filter_map[i][j] = True
            elif abs(i - img_height_half) < abs(i2 - img_height_half):
                filter_map[i][j] = True
    
    sum = 0
    for i in range(img_height):
        for j in range(img_width):
            if filter_map[i][j]:
                sum += 1

    return torch.tensor(filter_map)


def filter_keypoints3(pts_, desc_, filter_map, invert_mask=False):
    x_coords = pts_[:, 0].long()
    y_coords = pts_[:, 1].long()
    mask = filter_map[y_coords, x_coords]

    if invert_mask:
        mask = ~mask

    pts = pts_[mask]
    desc = desc_[:, mask]
    return pts, desc


def filter_keypoints(pts_, desc_, img_hw, invert_mask=False):
    img_height = img_hw[0]
    img_width = img_hw[1]

    theta = (pts_[:, 1] / img_height) * np.pi
    phi = (pts_[:, 0] / img_width) * 2 * np.pi
    x, y, z = spherical_to_cartesian(phi, theta)
    rad = np.pi / 2
    xx, yy, zz = rotate_yaw(x, y, z, rad)
    xx, yy, zz = rotate_pitch(xx, yy, zz, rad)
    xx, yy, zz = rotate_yaw(xx, yy, zz, rad)
    _, new_theta = cartesian_to_spherical(xx, yy, zz)
    new_i = (new_theta / np.pi *2 + 1) * img_height //2
    mask_y = abs(pts_[:, 1] - img_height // 2) < abs(new_i - img_height // 2) 
    mask_x = abs(pts_[:, 0] - img_width // 2) < img_width * 3 // 8
    mask = mask_y & mask_x
    if invert_mask:
        mask = ~mask

    pts = pts_[mask]
    desc = desc_.T[mask].T

    return pts, desc

def filter_keypoints3(pts_, desc_, img_hw, invert_mask=False):
    img_height = img_hw[0]
    img_width = img_hw[1]

    theta_values = (pts_[:, 1] / img_height) * np.pi
    phi_values = (pts_[:, 0] / img_width) * 2 * np.pi  # x座標を使用して経度を計算
    theta_std = 4
    phi_std = 4
    mask_theta = (torch.pi/theta_std <= theta_values) & (theta_values < (theta_std-1)*torch.pi/theta_std)
    mask_phi = (np.pi/phi_std <= phi_values) & (phi_values < (phi_std*2-1)*np.pi/phi_std)
    mask = mask_theta & mask_phi
    if invert_mask:
        mask = ~mask

    pts = pts_[mask]
    desc = desc_.T[mask].T

    return pts, desc


def add_offset_to_image(pts_, crop_start_xy):
    pts_[:, 0] += crop_start_xy[1]
    pts_[:, 1] += crop_start_xy[0]

    return pts_



def spherical_to_cartesian(phi, theta):
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return x, y, z


def cartesian_to_spherical(x, y, z):
    theta = np.arcsin(z)
    phi = np.arctan2(y, x)
    return phi, theta



def convert_coordinates_vectorized(tensor, image_size_hw, rad=-np.pi/2):
    h, w = image_size_hw
    w_half = w / 2
    h_half = h / 2

    x_coords = tensor[:, 0]
    y_coords = tensor[:, 1]

    phi = (x_coords - w_half) * np.pi * 2 / w
    theta = (y_coords - h_half) * np.pi / h
    
    x, y, z = spherical_to_cartesian(phi, theta)

    xx, yy, zz = rotate_yaw(x, y, z, rad)
    xx, yy, zz = rotate_pitch(xx, yy, zz, rad)
    xx, yy, zz = rotate_yaw(xx, yy, zz, rad)

    new_phi, new_theta = cartesian_to_spherical(xx, yy, zz)

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
