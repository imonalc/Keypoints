import torch
import kornia as K
import kornia.feature as KF
import cv2
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
loftr_matcher_indoor = KF.LoFTR(pretrained='indoor').to(device)
loftr_matcher_outdoor = KF.LoFTR(pretrained='outdoor').to(device)
OUTDOORS = ["urban1", "urban2", "urban3", "urban4"]

def loftr_match(path_o, path_r, scene):
    img1 = load_torch_image(path_o).to(device)
    img2 = load_torch_image(path_r).to(device)
    input_dict = {"image0": K.color.rgb_to_grayscale(img1), "image1": K.color.rgb_to_grayscale(img2)}
    matcher = loftr_matcher_outdoor if scene in OUTDOORS else loftr_matcher_indoor

    feature_time1 = time.perf_counter()
    correspondences = matcher(input_dict)
    feature_time2 = time.perf_counter()

    x1_ = correspondences["keypoints0"].cpu().numpy()
    x2_ = correspondences["keypoints1"].cpu().numpy()
    s_pts1 = x1_.copy()
    s_pts2 = x2_.copy()
    feature_time = feature_time2 - feature_time1

    return s_pts1, s_pts2, x1_, x2_, feature_time

def load_torch_image(fname):
    img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img