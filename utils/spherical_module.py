import cv2
import numpy as np
import time

from utils.ransac   import *
from utils.keypoint import *
from utils.metrics  import *
from utils.camera_recovering import *
from scipy.spatial import distance
import math
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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



def filter_middle_latitude(pts_, desc_, img_hw, invert_mask=False):
    img_height = img_hw[0]
    img_width = img_hw[1]

    theta_values = (pts_[:, 1] / img_height) * torch.pi
    phi_values = (pts_[:, 0] / img_width) * 2 * torch.pi
    theta_std = 6
    phi_std = 8
    mask_theta = (torch.pi/theta_std <= theta_values) & (theta_values < (theta_std-1)*torch.pi/theta_std)
    mask_phi = (torch.pi/phi_std <= phi_values) & (phi_values < (phi_std*2-1)*torch.pi/phi_std)
    mask = mask_theta & mask_phi
    if invert_mask:
        mask = ~mask

    pts = pts_[mask]
    desc = desc_.T[mask].T

    return pts, desc



def filter_keypoints(pts_, desc_, img_hw, invert_mask=False):
    img_height = img_hw[0]
    img_width = img_hw[1]

    theta = (pts_[:, 1] / img_height) * torch.pi
    phi = (pts_[:, 0] / img_width) * 2 * torch.pi
    x, y, z = spherical_to_cartesian(phi, theta)
    rad = torch.pi / 2
    xx, yy, zz = rotate_yaw(x, y, z, rad)
    xx, yy, zz = rotate_pitch(xx, yy, zz, rad)
    xx, yy, zz = rotate_yaw(xx, yy, zz, rad)
    _, new_theta = cartesian_to_spherical(xx, yy, zz)
    new_i = (new_theta / torch.pi * img_height + img_height) % img_height

    mask_y = torch.abs(pts_[:, 1] - img_height // 2) < torch.abs(new_i - img_height // 2) 
    mask_x = torch.abs(pts_[:, 0] - img_width // 2) < img_width * 3 // 8
    mask = mask_y & mask_x

    if invert_mask:
        mask = ~mask

    pts = pts_[mask]
    desc = desc_.T[mask].T

    return pts, desc



def filter_keypoints_abridged(pts_, desc_, img_hw, invert_mask=False):
    img_height = img_hw[0]
    img_width = img_hw[1]

    theta_values = (pts_[:, 1] / img_height) * torch.pi
    phi_values = (pts_[:, 0] / img_width) * 2 * torch.pi
    theta_std = 4
    phi_std = 4
    mask_theta = (torch.pi/theta_std <= theta_values) & (theta_values < (theta_std-1)*torch.pi/theta_std)
    mask_phi = (torch.pi/phi_std <= phi_values) & (phi_values < (phi_std*2-1)*torch.pi/phi_std)
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



def convert_coordinates_vectorized(tensor, image_size_hw, rad=-torch.pi/2):
    h, w = image_size_hw
    w_half = w / 2
    h_half = h / 2

    x_coords = tensor[:, 0]
    y_coords = tensor[:, 1]

    phi = (x_coords - w_half) * torch.pi * 2 / w
    theta = (y_coords - h_half) * torch.pi / h
    
    x, y, z = spherical_to_cartesian(phi, theta)

    xx, yy, zz = rotate_yaw(x, y, z, rad)
    xx, yy, zz = rotate_pitch(xx, yy, zz, rad)
    xx, yy, zz = rotate_yaw(xx, yy, zz, rad)

    new_phi, new_theta = cartesian_to_spherical(xx, yy, zz)

    new_y = new_theta * h_half / (torch.pi/2) + h_half
    new_x = new_phi * w_half / torch.pi + w_half
    
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



#### cube map ####
def create_equirectangular_map(input_w, input_h, output_sqr, displacement, direction='z'):
    row_indices, col_indices = np.meshgrid(np.arange(output_sqr), np.arange(output_sqr))
    x = y = z = np.zeros_like(row_indices, dtype=float)
    
    if direction == 'z':
        x = row_indices - output_sqr / 2.0
        y = col_indices - output_sqr / 2.0
        z += displacement
    elif direction == 'x':
        y = row_indices - output_sqr / 2.0
        z = col_indices - output_sqr / 2.0
        x += displacement
    elif direction == 'y':
        z = row_indices - output_sqr / 2.0
        x = col_indices - output_sqr / 2.0
        y += displacement

    rho = np.sqrt(x**2 + y**2 + z**2)
    norm_theta = np.arctan2(y, x) / (2 * np.pi)
    norm_phi = (np.pi - np.arccos(z / rho)) / np.pi
    ix = norm_theta * input_w
    iy = norm_phi * input_h

    ix = ix % input_w
    iy = iy % input_h

    return ix, iy


def convert_img_eq_to_cube(img, output_sqr=256, margin=50):
    normalized_f = 2.0
    displacement = output_sqr / normalized_f
    expanded_output_sqr = output_sqr + margin * 2
    input_h, input_w = img.shape[0], img.shape[1]

    make_map_time1 = time.perf_counter()
    bottom_map_x, bottom_map_y = create_equirectangular_map(
        input_w, input_h, expanded_output_sqr, displacement, 'z'
    )
    top_map_x, top_map_y = create_equirectangular_map(
        input_w, input_h, expanded_output_sqr, -displacement, 'z'
    )
    front_map_x, front_map_y = create_equirectangular_map(
        input_w, input_h, expanded_output_sqr, -displacement, 'x'
    )
    back_map_x, back_map_y = create_equirectangular_map(
        input_w, input_h, expanded_output_sqr, displacement, 'x'
    )
    left_map_x, left_map_y = create_equirectangular_map(
        input_w, input_h, expanded_output_sqr, -displacement, 'y'
    )
    right_map_x, right_map_y = create_equirectangular_map(
        input_w, input_h, expanded_output_sqr, displacement, 'y'
    )
    make_map_time2 = time.perf_counter()

    remap_time1 = time.perf_counter()
    bottom_img = cv2.remap(
        img,
        bottom_map_x.astype("float32"),
        bottom_map_y.astype("float32"),
        cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP,
    )
    bottom_img = cv2.rotate(bottom_img, cv2.ROTATE_90_CLOCKWISE)
    
    top_img = cv2.remap(
        img, top_map_x.astype("float32"), top_map_y.astype("float32"), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP
    )
    top_img = cv2.flip(top_img, 1)
    top_img = cv2.rotate(top_img, cv2.ROTATE_90_CLOCKWISE)

    front_img = cv2.remap(
        img,
        front_map_x.astype("float32"),
        front_map_y.astype("float32"),
        cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP,
    )
    front_img = cv2.flip(front_img, 1)


    back_img = cv2.remap(
        img, back_map_x.astype("float32"), back_map_y.astype("float32"), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP
    )
    
    left_img = cv2.remap(
        img, left_map_x.astype("float32"), left_map_y.astype("float32"), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP
    )
    left_img = cv2.flip(left_img, 1)
    left_img = cv2.rotate(left_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    right_img = cv2.remap(
        img,
        right_map_x.astype("float32"),
        right_map_y.astype("float32"),
        cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP,
    )
    right_img = cv2.rotate(right_img, cv2.ROTATE_90_CLOCKWISE)
    remap_time2 = time.perf_counter()

    make_map_time = make_map_time2 - make_map_time1
    remap_time = remap_time2 - remap_time1

    return [back_img, bottom_img, front_img, left_img, right_img, top_img], make_map_time, remap_time


def cube_coord_to_3d_vector(face, cor_xy, width):
    x, y = cor_xy
    x -= width
    y -= width

    if face == "front":
        return torch.tensor([width, y, -x])
    elif face == "back":
        return torch.tensor([-width, -y, -x])
    elif face == "left":
        return torch.tensor([-y, width, -x])
    elif face == "right":
        return torch.tensor([y, -width, -x])
    elif face == "top":
        return torch.tensor([x, y, width])
    elif face == "bottom":
        return torch.tensor([-x, y, -width])
    else:
        raise ValueError(f"Invalid face name: {face}")
    
def vector_to_equirectangular_coord(vec, width):
    r = torch.sqrt(torch.sum(vec**2))
    theta = torch.acos(vec[2] / r) # polar angle (0 <= theta <= pi)
    phi = torch.atan2(vec[1], vec[0]) # azimuthal angle (-pi <= phi <= pi)
    
    x = width * 8 * (phi + torch.pi) / (2 * torch.pi)
    y = width * 4 * theta / torch.pi
    
    return x.item(), y.item()

def cube_to_equirectangular_coord(face, cor_xy, width):
    vec = cube_coord_to_3d_vector(face, cor_xy, width)
    return vector_to_equirectangular_coord(vec, width)


def cube_coords_to_3d_vectors(face, coords, width):
    x, y = coords[:, 0], coords[:, 1]
    x -= width
    y -= width

    face_dict = {
        "front": torch.stack([width * torch.ones_like(x), y, -x], dim=-1),
        "back": torch.stack([-width * torch.ones_like(x), -y, x], dim=-1),
        "left": torch.stack([-y, width * torch.ones_like(x), -x], dim=-1),
        "right": torch.stack([y, -width * torch.ones_like(x), -x], dim=-1),
        "top": torch.stack([x, y, width * torch.ones_like(x)], dim=-1),
        "bottom": torch.stack([-x, y, -width * torch.ones_like(x)], dim=-1)
    }
    return face_dict[face]


def vectors_to_equirectangular_coords(vectors, width):
    r = torch.sqrt(torch.sum(vectors**2, dim=1))
    theta = torch.acos(vectors[:, 2] / r) # polar angle
    phi = torch.atan2(vectors[:, 1], vectors[:, 0]) # azimuthal angle
    
    x = width * 8 * (phi + torch.pi) / (2 * torch.pi)
    y = width * 4 * theta / torch.pi
    
    return torch.stack([x, y], dim=-1)


def batch_cube_to_equirectangular(face, coords, width):
    vecs = cube_coords_to_3d_vectors(face, coords, width //2)
    return vectors_to_equirectangular_coords(vecs, width //2)