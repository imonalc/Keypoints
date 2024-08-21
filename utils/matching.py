import cv2

import pandas as pd
import numpy as np


from utils.ransac   import *
from utils.keypoint import *
from utils.metrics  import *
from utils.camera_recovering import *
from scipy.spatial import distance

from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue

OUTDOORS = ["urban1", "urban2", "urban3", "urban4"]


class SGMatching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SuperGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))
        self.superglue = SuperGlue(config.get('superglue', {}))

    def forward(self, data):
        """ Run SuperPoint (optionally) and SuperGlue
        SuperPoint is skipped if ['keypoints0', 'keypoints1'] exist in input
        Args:
          data: dictionary with minimal keys: ['image0', 'image1']
        """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        matching_b = time.perf_counter()
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        pred = {**pred, **self.superglue(data)}
        matching_a = time.perf_counter()
        matching_time = matching_a - matching_b

        return pred, matching_time

config_indoor = {
    'superglue': {
        'weights': "indoor",
        'match_threshold': 0.03
    }
}
sg_matcher_indoor = SGMatching(config_indoor).eval().to(device)
config_outdoor = {
    'superglue': {
        'weights': "outdoor",
        'match_threshold': 0.03
    }
}
sg_matcher_outdoor = SGMatching(config_outdoor).eval().to(device)


def sort_key(pts1, pts2, desc1, desc2, points):

    ind1 = np.argsort(pts1[:,2].numpy(),axis = 0)[::-1]
    ind2 = np.argsort(pts2[:,2].numpy(),axis = 0)[::-1]

    max1 = np.min([points,ind1.shape[0]])
    max2 = np.min([points,ind2.shape[0]])

    ind1 = ind1[:max1]
    ind2 = ind2[:max2]

    pts1 = pts1[ind1.copy(),:]
    pts2 = pts2[ind2.copy(),:]

    scores1 = pts1[:, 2:3].clone()
    scores2 = pts2[:, 2:3].clone()

    desc1 = desc1[:,ind1.copy()]
    desc2 = desc2[:,ind2.copy()]

    pts1 = np.concatenate((pts1[:,:2], np.ones((pts1.shape[0],1))), axis = 1 )
    pts2 = np.concatenate((pts2[:,:2], np.ones((pts2.shape[0],1))), axis = 1 )

    desc1 = np.transpose(desc1,[1,0]).numpy()
    desc2 = np.transpose(desc2,[1,0]).numpy()

    return pts1, pts2, desc1, desc2, scores1, scores2


def sort_key_div(pts1, desc1, points):
    ind1 = np.argsort(pts1[:,2].numpy(),axis = 0)[::-1]
    max1 = np.min([points,ind1.shape[0]])
    ind1 = ind1[:max1]
    pts1 = pts1[ind1.copy(),:]
    scores1 = pts1[:, 2:3].clone()
    desc1 = desc1[:,ind1.copy()]
    pts1 = np.concatenate((pts1[:,:2], np.ones((pts1.shape[0],1))), axis = 1 )
    desc1 = np.transpose(desc1,[1,0]).numpy()

    return pts1, desc1, scores1,


def bfknn_matcher(s_desc1, s_desc2, distance_eval, constant):
    thresh = constant
    bf = cv2.BFMatcher(distance_eval, False)
    matches = bf.knnMatch(s_desc1,s_desc2, k=2)
    good = []
    for m,n in matches:
        if m.distance < thresh * n.distance:
            good.append(m)
    matches = good

    return matches



def flannknn_matcher(s_desc1, s_desc2, distance_eval, constant):
    thresh = 0.95
    index_params = dict(algorithm=distance_eval, trees=5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(s_desc1, s_desc2, k=2)
    good = []
    for m,n in matches:
        if m.distance < thresh * n.distance:
            good.append(m)
    matches = good

    return matches



def mnn_matcher_aliked(desc1, desc2):
    sim = desc1 @ desc2.transpose()
    sim[sim < 0.75] = 0
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids1 = np.arange(0, sim.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]])
    matches_idx = matches.transpose()
    matchesD = [cv2.DMatch(i, j, 0) for i, j in matches_idx]
    
    return matchesD



def hamming_distance_optimized(desc1, desc2):
    return np.bitwise_xor(desc1[:, None, :], desc2).sum(axis=-1)


def hamming_distance(x, y):
    return np.sum(x != y, axis=-1)


def mnn_matcher_hamming(desc1, desc2, constant_cut=0.0):
    desc1 = desc1.astype(np.uint8)
    desc2 = desc2.astype(np.uint8)
    desc1 = np.unpackbits(desc1, axis=1)
    desc2 = np.unpackbits(desc2, axis=1)
    

    distances = hamming_distance_optimized(desc1, desc2)
    nn12 = np.argmin(distances, axis=1)
    nn21 = np.argmin(distances, axis=0)
    ids1 = np.arange(desc1.shape[0])
    mask = (ids1 == nn21[nn12])
    
    matches = np.stack([ids1[mask], nn12[mask]], axis=1)
    matched_distances = distances[ids1[mask], nn12[mask]]

    matches_with_distances = np.hstack([matches, matched_distances[:, np.newaxis]])
    sorted_matches_with_distances = matches_with_distances[matches_with_distances[:, 2].argsort()]

    if constant_cut == 0:
        matches = sorted_matches_with_distances[:, :2].transpose().astype(int)
    else:
        num_to_remove = int(len(sorted_matches_with_distances) * constant_cut)
        matches = sorted_matches_with_distances[:-num_to_remove, :2].transpose().astype(int)
    
    matches_idx = matches.transpose()
    matches = [cv2.DMatch(i, j, 0) for i, j in matches_idx]

    return matches


def mnn_matcher_alike(desc1, desc2):
    sim = desc1 @ desc2.transpose()
    sim[sim < 0.7] = 0
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids1 = np.arange(0, sim.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]])
    matches_idx = matches.transpose()
    matches = [cv2.DMatch(i, j, 0) for i, j in matches_idx]

    return matches


def mnn_matcher_L2(desc1, desc2, constant_cut=0.0):
    d1_square = np.sum(np.square(desc1), axis=1, keepdims=True)
    d2_square = np.sum(np.square(desc2), axis=1, keepdims=True)
    distances = np.sqrt(d1_square - 2 * np.dot(desc1, desc2.T) + d2_square.T)
    distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

    nn12 = np.argmin(distances, axis=1)
    nn21 = np.argmin(distances, axis=0)
    ids1 = np.arange(0, distances.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]])
    matched_distances = distances[ids1[mask], nn12[mask]]

    matches_with_distances = np.hstack([matches.transpose(), matched_distances[:, np.newaxis]])
    sorted_matches_with_distances = matches_with_distances[matches_with_distances[:, 2].argsort()]

    if constant_cut == 0:
        matches = sorted_matches_with_distances[:, :2].transpose().astype(int)
    else:
        num_to_remove = int(len(sorted_matches_with_distances) * constant_cut)
        matches = sorted_matches_with_distances[:-num_to_remove, :2].transpose().astype(int)
    matches_idx = matches.transpose()
    matches = [cv2.DMatch(i, j, 0) for i, j in matches_idx]

    return matches



def matched_points(pts1, pts2, desc1, desc2, opt, args_opt, match="BF", constant=0):
    if opt[-1] == 'p':
        porce = int(opt[:-1])
        n_key = int(porce/100 * pts1.shape[0])
    else:
        n_key = int(opt)
    
    s_pts1  = pts1.copy()[:n_key,:]
    s_pts2  = pts2.copy()[:n_key,:]
    s_desc1 = desc1.copy().astype('float32')[:n_key,:]
    s_desc2 = desc2.copy().astype('float32')[:n_key,:]


    if 'args_opt' in ['orb', 'sphorb', 'akaze']:
        s_desc1 = s_desc1.astype(np.uint8)
        s_desc2 = s_desc2.astype(np.uint8)
        distance_eval = cv2.NORM_HAMMING
        distance_eval_FLANN = 6
    else:
        distance_eval = cv2.NORM_L2
        distance_eval_FLANN = 1
    
    if "aliked" in args_opt:
        matches = mnn_matcher_aliked(s_desc1, s_desc2)
    elif match == 'BF':
        bf = cv2.BFMatcher(distance_eval, True)
        matches = bf.match(s_desc1, s_desc2)
    elif match == 'BF_KNN':
        if args_opt == "superpoint":
            constant = 0.95
        elif args_opt in ["orb", "sphorb", 'akaze']:
            constant = 0.9
        elif args_opt == "sift":
            constant = 0.75
        matches = bfknn_matcher(s_desc1, s_desc2, distance_eval, constant)
    elif match == 'FLANN_KNN':
        matches = flannknn_matcher(s_desc1, s_desc2, distance_eval_FLANN, constant)
    elif match == 'MNN':
        if 'args_opt' in ['orb', 'sphorb', 'akaze']:
            matches = mnn_matcher_hamming(desc1, desc2)
        elif 'args_opt' in ['alike', 'aliked']:
            matches = mnn_matcher_alike(desc1, desc2)
        else:
            matches = mnn_matcher_L2(s_desc1, s_desc2)
    else:
        raise ValueError("Invalid matching method specified.")
    
    M = np.zeros((2,len(matches)))
    for ind, match in zip(np.arange(len(matches)),matches):
        M[0,ind] = match.queryIdx
        M[1,ind] = match.trainIdx

    return s_pts1, s_pts2, s_pts1[M[0,:].astype(int),:3], s_pts2[M[1,:].astype(int),:3]



def get_error(x1, x2, Rx, Tx):

    S = computeEssentialMatrixByRANSAC(x1, x2)
    I = S[1]
    I = I.astype(np.int64)

    x1 = x1[I,:]
    x2 = x2[I,:]

    F = calc_ematrix(x1,x2)


    R1,R2,T1,T2 = decomposeE(F)

    R,T = choose_rt(R1,R2,T1,T2,x1,x2)

    R_error = r_error(Rx,R)
    T_error = t_error(Tx,T)

    return R_error, T_error



def get_descriptor(descriptor):
    descriptor_main = descriptor
    image_mode = 'erp'
    if descriptor[0] == 't':
        descriptor_main = descriptor_main[1:]
        image_mode = 'tangent'
    elif descriptor[0] == 'c':
        descriptor_main = descriptor_main[1:]
        image_mode = 'cube'
    elif descriptor[0] == 'p':
        descriptor_main = descriptor_main[1:]
        image_mode = 'proposed'
    elif descriptor[0] == 'r':
        descriptor_main = descriptor_main[1:]
        image_mode = 'rotated'
    
    descriptor_configs = {
        'sphorb': ('sphorb', image_mode, 640),
        'loftr': ('loftr', image_mode, 512),
        'spglue': ('superpoint', image_mode, 512),
        'sift': ('sift', image_mode, 512),
        'orb': ('orb', image_mode, 512),
        'spoint': ('superpoint', image_mode, 512),
        'akaze': ('akaze', image_mode, 512),
        'alike': ('alike', image_mode, 512),
        'aliked': ('aliked', image_mode, 512),
    }

    return descriptor_configs.get(descriptor_main, ('unknown', 'unknown', 0,))



def adjust_vertical_intensity(pts1_, desc1_, img_shape):
    img_height = img_shape[0]
    center_y = img_height / 2.0

    for i, pt in enumerate(pts1_):
        distance_from_center_y = abs(pt[1] - center_y)
        adjustment_factor = np.sqrt(1 - (distance_from_center_y / center_y)**2)
        if distance_from_center_y == center_y:
            adjustment_factor = 1 - 1

        pts1_[i, 2] *= adjustment_factor
    
    return pts1_, desc1_



def AUC(ROT, TRA, MET, L):

    RAUC  = np.zeros(len(L))
    TAUC  = np.zeros(len(L))

    for index, t in enumerate(L):
        ids = np.where(ROT<np.radians(t))[0]
        RAUC[index] = len(ids)/len(ROT)

    for index, t in enumerate(L):
        ids = np.where(TRA<np.radians(t))[0]
        TAUC[index] = len(ids)/len(TRA)

    return RAUC, TAUC, np.array(MET)


def superglue_matching(path_o, path_r, scale_factor, device, scene):
    x1_, x2_ = [], []
    matcher = sg_matcher_outdoor if scene in OUTDOORS else sg_matcher_indoor
    img_o = load_torch_img(path_o)[:3, ...].float()
    img_o = F.interpolate(img_o.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
    img_r = load_torch_img(path_r)[:3, ...].float()
    img_r = F.interpolate(img_r.unsqueeze(0), scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
    img_o = torch2numpy(img_o.byte())
    img_r = torch2numpy(img_r.byte())
    img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    img_o_tensor = torch.from_numpy(img_o).float()/255.0
    img_o_tensor = img_o_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    img_r_tensor = torch.from_numpy(img_r).float()/255.0
    img_r_tensor = img_r_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    img_o_tensor = img_o_tensor[:, 0:1, :, :]
    img_r_tensor = img_r_tensor[:, 0:1, :, :]
    
    data = {
        "image0":img_o_tensor,
        "image1":img_r_tensor,
    }
    
    pred, matching_time = matcher(data)
    keypoints0_tensor = pred["keypoints0"][0]
    keypoints1_tensor = pred["keypoints1"][0]

    kpt1 = keypoints0_tensor.cpu().numpy()
    kpt1 = np.hstack((kpt1, np.ones((kpt1.shape[0], 1))))
    kpt2 = keypoints1_tensor.cpu().numpy()
    kpt2 = np.hstack((kpt2, np.ones((kpt2.shape[0], 1))))
    matches1 = pred["matches0"][0].cpu().numpy()
    for i in range(len(matches1)):
        if matches1[i] == -1: continue
        x1_.append(kpt1[i])
        x2_.append(kpt2[matches1[i]])
    s_pts1, s_pts2 = np.array(kpt1), np.array(kpt2)
    x1_, x2_ = np.array(x1_), np.array(x2_)

    return s_pts1, s_pts2, x1_, x2_, matching_time


def get_data(DATAS):
    if len(DATAS) == 1:
        data = DATAS[0]
    elif set(['Urban1','Urban2','Urban3','Urban4']) == set(DATAS):
        data = 'Outdoor'
    elif set(['Realistic','Interior1','Interior2','Room','Classroom']) == set(DATAS):
        data = 'Indoor'
    elif set(['Urban1_R','Urban2_R','Urban3_R','Urban4_R','Realistic_R','Interior1_R','Interior2_R','Room_R','Classroom_R']) == set(DATAS):
        data = 'OnlyRot'
    elif set(['Urban1_T','Urban2_T','Urban3_T','Urban4_T','Realistic_T','Interior1_T','Interior2_T','Room_T','Classroom_T']) == set(DATAS):
        data = 'OnlyTra'
    else:
        data = ''
        for DA in DATAS:
            data+=DA

    return data



def get_kd(array):

    array = np.array(array)
    delimiter = int(array[-1])
    A = array[:-1]
    K = A[:delimiter].reshape(-1,3)
    D = A[delimiter:].reshape(-1,32)
    return K,D