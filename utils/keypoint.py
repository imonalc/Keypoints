import torch
import torch.nn.functional as F
from spherical_distortion.functional import create_tangent_images, create_sample_map, create_tangent_images_from_map
from spherical_distortion.util import *
import time
import cv2
import numpy as np


from utils.spherical_module import *
import utils.superpoint.magic_sp.superpoint as magic_sp
import utils.superpoint.train_sp.superpoint as train_sp
from utils.ALIKE.alike import ALike, configs
from utils.ALIKED.nets.aliked import ALIKED

padding_length = 50

## call instance
orb_model = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE, nfeatures=1000)
sift_model = cv2.SIFT_create(nfeatures=1000)
akaze_model = cv2.AKAZE_create(threshold=0.0001)
sp_model = train_sp.SuperPointFrontend(weights_path = 'utils/models/superpoint-trained.pth.tar', 
                                 nms_dist= 4, #nms_dist, 
                                 conf_thresh = 0.01, #conf_thresh, 
                                 nn_thresh= 0.7, #nn_thresh, 
                                 cuda = True ) # 4 0.015 or 3 0.005
aliked_model = ALIKED(model_name="aliked-n16",
    device="cuda",
    top_k=-1,
    scores_th=0.15,
    n_limit=1000)





def process_img(img):
    """ Process a image transposing it and convert to grayscale format, Then normalize

    img: 3 x H x W

    """
    img = np.transpose(img.numpy(),[1,2,0])
    H, W = img.shape[0], img.shape[1]
    grayim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #interp = cv2.INTER_AREA
    #grayim = cv2.resize(grayim, (160, 120), interpolation=interp)
    grayim = (grayim.astype('float32') / 255.)
    return grayim



def computes_superpoint_keypoints(img):
    pts, desc, _ = sp_model.run(img)
    kpt_details = np.zeros((pts.shape[1],4))
    kpt_details[:,0] = pts[0,:]
    kpt_details[:,1] = pts[1,:]
    kpt_details[:,2] = pts[2,:]
    kpt_details[:,3] = pts[2,:]
    if pts.shape[1] != 0:
        desc = np.transpose(desc, [1,0])
        return torch.from_numpy(kpt_details), torch.from_numpy(desc)
    return None



def computes_alike_keypoints(img, model_nm="alike-n", device="cuda", top_k=-1, scores_th=0.2, n_limit=10000):
    model = ALike(**configs[model_nm],
        device=device,
        top_k=top_k,
        scores_th=scores_th,
        n_limit=n_limit)
    img_rgb = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
    pred = model(img_rgb, sub_pixel=True)
    kpts = pred["keypoints"]
    desc = pred["descriptors"]
    scores = pred["scores"]
    score_map = pred["scores_map"]
    #print(kpts.shape, desc.shape, scores.shape, score_map.shape)

    kpt_details = np.zeros((kpts.shape[0],4))

    kpt_details[:,0] = kpts[:,0]
    kpt_details[:,1] = kpts[:,1]
    kpt_details[:,2] = scores.squeeze()
    kpt_details[:,3] = scores.squeeze()

    if len(kpts)>0:
        #desc = np.transpose(desc, [1,0])
        return torch.from_numpy(kpt_details), torch.from_numpy(desc)
    return None



def computes_aliked_keypoints(img):
    img_rgb = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB)
    pred = aliked_model.run(img_rgb)
    kpts = pred["keypoints"]
    desc = pred["descriptors"]
    scores = pred["scores"]
    #print(kpts.shape, desc.shape, scores.shape, score_map.shape)

    kpt_details = np.zeros((kpts.shape[0],4))

    kpt_details[:,0] = kpts[:,0]
    kpt_details[:,1] = kpts[:,1]
    kpt_details[:,2] = scores.squeeze()
    kpt_details[:,3] = scores.squeeze()

    if len(kpts)>0:
        #desc = np.transpose(desc, [1,0])
        return torch.from_numpy(kpt_details), torch.from_numpy(desc)
    return None


def format_keypoints(keypoints, desc):
    """
    Formatear puntos de interes y descriptores de opencv para su posterior tratamiento

    """
    coords = torch.tensor([kp.pt for kp in keypoints])
    responsex = torch.tensor([kp.response for kp in keypoints])
    responsey = torch.tensor([kp.response for kp in keypoints])
    desc = torch.from_numpy(desc)
    return torch.cat((coords, responsex.unsqueeze(1), responsey.unsqueeze(1)), -1), desc


def computes_orb_keypoints(img):
    img = torch2numpy(img.byte())
    keypoints, desc = orb_model.detectAndCompute(img, None)

    # Keypoints is a list of lenght N, desc is N x 128
    if len(keypoints) > 0:
        return format_keypoints(keypoints, desc)
    return None


def computes_akaze_keypoints(img):
    img = torch2numpy(img.byte())
    keypoints, descriptors = akaze_model.detectAndCompute(img, None)

    if len(keypoints) > 0:
        return format_keypoints(keypoints, descriptors)
    return None


def computes_surf_keypoints(img):

    img = torch2numpy(img.byte())

    # Initialize OpenCV ORB detector
    surf = cv2.xfeatures2d.SURF_create(nfeatures=10000)

    keypoints, desc = surf.detectAndCompute(img, None)

    # Keypoints is a list of lenght N, desc is N x 128
    if len(keypoints) > 0:
        return format_keypoints(keypoints, desc)
    return None


def computes_sift_keypoints(img):
    img = torch2numpy(img.byte())
    keypoints, desc = sift_model.detectAndCompute(img, None)

    # Keypoints is a list of lenght N, desc is N x 128
    if len(keypoints) > 0:
        return format_keypoints(keypoints, desc)
    return None


def compute_keypoints(img, opt):
    feature_time1 = time.perf_counter()
    if opt == 'superpoint':
        img = process_img(img)
        ret_img = computes_superpoint_keypoints(img)
    if opt == 'sift':
        ret_img = computes_sift_keypoints(img)
    if opt == 'orb':
        ret_img = computes_orb_keypoints(img)
    if opt == 'surf':
        ret_img = computes_surf_keypoints(img)
    if opt == 'alike':
        ret_img = computes_alike_keypoints(img)
    if opt == 'aliked':
        ret_img = computes_aliked_keypoints(img)
    if opt == 'akaze':
        ret_img = computes_akaze_keypoints(img)
    feature_time2 = time.perf_counter()
    feature_time = feature_time2 - feature_time1

    return ret_img, feature_time



def keypoint_tangent_images(tex_image, base_order, sample_order, image_shape, opt = 'superpoint', crop_degree=0):
    """
    Extracts only the visible Superpoint features from a collection tangent image. That is, only returns the keypoints visible to a spherical camera at the center of the icosahedron.

    tex_image: 3 x N x H x W
    corners: N x 4 x 3 coordinates of tangent image corners in 3D
    image_shape: (H, W) of equirectangular image that we render back to
    crop_degree: [optional] scalar value in degrees dictating how much of input equirectangular image is 0-padding

    returns [visible_kp, visible_desc] (M x 4, M x length_descriptors)
    """

    # ----------------------------------------------
    # Compute descriptors for each patch
    # ----------------------------------------------
    kp_list = []  # Stores keypoint coords
    desc_list = []  # Stores keypoint descriptors
    feature_time = 0
    for i in range(tex_image.shape[1]):
        img = tex_image[:, i, ...]
        kp_details, fp_time = compute_keypoints(img, opt)
        feature_time += fp_time

        if kp_details is not None:
            valid_mask = get_valid_coordinates(base_order,
                                               sample_order,
                                               i,
                                               kp_details[0][:, :2],
                                               return_mask=True)[1]
            visible_kp = kp_details[0][valid_mask]
            visible_desc = kp_details[1][valid_mask]

            # Convert tangent image coordinates to equirectangular
            visible_kp[:, :2] = convert_spherical_to_image(
                torch.stack(
                    convert_tangent_image_coordinates_to_spherical(
                        base_order, sample_order, i, visible_kp[:, :2]), -1),
                image_shape)

            kp_list.append(visible_kp)
            desc_list.append(visible_desc)
        

    all_visible_kp = torch.cat(kp_list, 0).float()  # M x 4 (x, y, s, o)
    all_visible_desc = torch.cat(desc_list, 0).float()  # M x 128

    # If top top and bottom of image is padding
    crop_h = compute_crop(image_shape, crop_degree)

    # Ignore keypoints along the stitching boundary
    mask = (all_visible_kp[:, 1] > crop_h) & (all_visible_kp[:, 1] <
                                              image_shape[0] - crop_h)
    all_visible_kp = all_visible_kp[mask]  # M x 4
    all_visible_desc = all_visible_desc[mask]  # M x 128

    return all_visible_kp, np.transpose(all_visible_desc,[1,0]), feature_time



def keypoint_cube_images(img, opt, output_sqr=256, margin=50):
    kp_list, desc_list = [], []
    [back_img, bottom_img, front_img, left_img, right_img, top_img], make_map_time, remap_time = convert_img_eq_to_cube(img.permute(1, 2, 0).cpu().numpy(), output_sqr, margin)

    face_dict = {"top": top_img,
                 "left": left_img,
                 "front": front_img,
                 "right": right_img,
                 "bottom": bottom_img, 
                 "back": back_img
    }
    feature_time = 0
    for idx, (face, img) in enumerate(face_dict.items()):
        img = torch.from_numpy(img.astype(np.float32)).clone().permute(2, 1, 0)
        #feature_time1 = time.perf_counter()
        kp_details, fp_time = compute_keypoints(img, opt)
        feature_time += fp_time
        
        if kp_details is not None:
            kp, desc = kp_details
            coords = kp[:, :2].clone()
            coords -= margin
            valid_mask = (coords >= 0) & (coords < output_sqr)
            valid_indices = valid_mask.all(dim=1)
            coords = coords[valid_indices]
            
            new_coords = batch_cube_to_equirectangular(face, coords, output_sqr)
            new_kps = kp[valid_indices, 2:]
            new_desc = desc[valid_indices]
            kp_converted = torch.cat([new_coords, new_kps], dim=1)
            kp_list.append(kp_converted)
            desc_list.append(new_desc)
        #feature_time2 = time.perf_counter()
        #feature_time += feature_time2 - feature_time1
    
    #feature_time1 = time.perf_counter()
    kp_list = torch.cat(kp_list, dim=0)
    desc_list = torch.cat(desc_list, dim=0)
    #feature_time2 = time.perf_counter()

    #feature_time += feature_time2 - feature_time1

    return kp_list, np.transpose(desc_list,[1,0]), make_map_time, remap_time, feature_time



def keypoint_proposed(img, opt, scale_factor, img_hw):
    Y_remap, X_remap, make_map_time = make_image_map(img_hw)
    img_hw_crop = (img_hw[0]//2+padding_length*2, img_hw[1]*3//4+padding_length*2)
    crop_start_xy = ((img_hw[0]-img_hw_crop[0])//2 - 1, (img_hw[1]-img_hw_crop[1])//2 - 1)

    img1, img2, remap_time = remap_crop_image(img, (Y_remap, X_remap), img_hw_crop, crop_start_xy)

    img1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0)
    img1 = F.interpolate(img1, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0)
    img2 = F.interpolate(img2, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)

    #feature_time_b = time.perf_counter()
    pts1_, desc1_, feature_time1 = keypoint_equirectangular(img1, opt)
    pts2_, desc2_, feature_time2 = keypoint_equirectangular(img2, opt)
    feature_time = feature_time1 + feature_time2

    pts1_ = add_offset_to_image(pts1_, crop_start_xy)
    pts2_ = add_offset_to_image(pts2_, crop_start_xy)
    pts2_ = convert_coordinates_vectorized(pts2_, img_hw)
    pts1_, desc1_ = filter_keypoints(pts1_, desc1_, img_hw)
    pts2_, desc2_ = filter_keypoints(pts2_, desc2_, img_hw, invert_mask=True)
    image_kp = torch.cat((pts1_, pts2_), dim=0)
    image_desc = torch.cat((desc1_, desc2_), dim=1)
    #feature_time_a = time.perf_counter()
    #feature_time = feature_time_a - feature_time_b

    return image_kp, image_desc, make_map_time, remap_time, feature_time



def keypoint_equirectangular(img, opt ='superpoint', crop_degree=0):
    """
    img: torch style (C x H x W) torch tensor
    crop_degree: [optional] scalar value in degrees dictating how much of input equirectangular image is 0-padding

    returns [erp_kp, erp_desc] (M x 4, M x number_descriptors)
    """

    # ----------------------------------------------
    # Compute descriptors on equirect image
    # ----------------------------------------------
    #feature_time1 = time.perf_counter()
    feature_time = 0
    kp_details, fp_time = compute_keypoints(img, opt)
    feature_time += fp_time

    erp_kp = kp_details[0]
    erp_desc = kp_details[1]

    # If top top and bottom of image is padding
    crop_h = compute_crop(img.shape[-2:], crop_degree)

    # Ignore keypoints along the stitching boundary
    mask = (erp_kp[:, 1] > crop_h) & (erp_kp[:, 1] < img.shape[1] - crop_h)
    erp_kp = erp_kp[mask]
    erp_desc = erp_desc[mask]
    #feature_time2 = time.perf_counter()

    #feature_time = feature_time2 - feature_time1

    return erp_kp, np.transpose(erp_desc,[1,0]), feature_time


def keypoint_rotated(img, opt, scale_factor, img_hw):
    Y_remap1, X_remap1, Y_remap2, X_remap2, make_map_time = make_image_map_r(img_hw)
    img_hw_crop = (img_hw[0]//3+padding_length*2, img_hw[1])
    crop_start_xy = ((img_hw[0]-img_hw_crop[0])//2 - 1,  0)

    img1, img2, remap_time1 = remap_crop_image(img, (Y_remap1, X_remap1), img_hw_crop, crop_start_xy)
    _, img3, remap_time2 = remap_crop_image(img, (Y_remap2, X_remap2), img_hw_crop, crop_start_xy)
    #cv2.imwrite('img1.jpg', img1)
    #cv2.imwrite('img2.jpg', img2)
    #cv2.imwrite('img3.jpg', img3)
    remap_time = remap_time1 + remap_time2
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0)
    img1 = F.interpolate(img1, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0)
    img2 = F.interpolate(img2, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
    img3 = torch.from_numpy(img3).permute(2, 0, 1).float().unsqueeze(0)
    img3 = F.interpolate(img3, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
    
    #feature_time_b = time.perf_counter()
    pts1_, desc1_, feature_time1 = keypoint_equirectangular(img1, opt)
    pts2_, desc2_, feature_time2 = keypoint_equirectangular(img2, opt)
    pts3_, desc3_, feature_time3 = keypoint_equirectangular(img3, opt)
    feature_time = feature_time1 + feature_time2 + feature_time3

    pts1_ = add_offset_to_image(pts1_, crop_start_xy)
    pts2_ = add_offset_to_image(pts2_, crop_start_xy)
    pts3_ = add_offset_to_image(pts3_, crop_start_xy)

    pts1_, desc1_ = filter_keypoints_r(pts1_, desc1_, img_hw)
    pts2_, desc2_ = filter_keypoints_r(pts2_, desc2_, img_hw)
    pts3_, desc3_ = filter_keypoints_r(pts3_, desc3_, img_hw)

    pts2_ = convert_coordinates_vectorized_r(pts2_, img_hw, rad=-torch.pi*2/3)
    pts3_ = convert_coordinates_vectorized_r(pts3_, img_hw, rad=torch.pi*2/3)
    image_kp = torch.cat([pts1_, pts2_, pts3_], dim=0)
    image_desc = torch.cat([desc1_, desc2_, desc3_], dim=1)
    #feature_time_a = time.perf_counter()
    #feature_time = feature_time_a - feature_time_b

    return image_kp, image_desc, make_map_time, remap_time, feature_time


def process_image_to_keypoints(image_path, scale_factor, base_order, sample_order, opt, mode, img_hw):
    img = load_torch_img(image_path)[:3, ...].float()
    img = F.interpolate(img.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
    make_map_time, remap_time, feature_time = 0, 0, 0

    if mode == 'tangent':
        time1 = time.perf_counter()
        sample_map = create_sample_map(img, base_order,sample_order)
        time2 = time.perf_counter()
        tex_image = create_tangent_images_from_map(img, sample_map, base_order, sample_order).byte()
        time3 = time.perf_counter()
        image_kp, image_desc, feature_time = keypoint_tangent_images(tex_image, base_order, sample_order, img.shape[-2:], opt , 0)
        time4 = time.perf_counter()
        
        make_map_time = time2 - time1
        remap_time = time3 - time2
    elif mode == 'cube':
        image_kp, image_desc, make_map_time, remap_time, feature_time = keypoint_cube_images(img, opt)
    elif mode == 'proposed':
        img = cv2.imread(image_path)
        image_kp, image_desc, make_map_time, remap_time, feature_time = keypoint_proposed(img, opt, scale_factor, img_hw)
    elif mode == 'erp':
        image_kp, image_desc, feature_time = keypoint_equirectangular(img, opt)
    elif mode == 'rotated':
        img = cv2.imread(image_path)
        image_kp, image_desc, make_map_time, remap_time, feature_time = keypoint_rotated(img, opt, scale_factor, img_hw)

    return image_kp, image_desc, make_map_time, remap_time, feature_time


