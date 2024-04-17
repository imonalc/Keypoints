import cv2

import pandas as pd
import numpy as np


from utils.ransac   import *
from utils.keypoint import *
from utils.metrics  import *
from utils.camera_recovering import *
from scipy.spatial import distance



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


def hamming_distance_optimized(desc1, desc2):
    return np.bitwise_xor(desc1[:, None, :], desc2).sum(axis=-1)


def hamming_distance(x, y):
    return np.sum(x != y, axis=-1)

def mnn_matcher_hamming(desc1, desc2, constant_cut):
    desc1 = np.unpackbits(desc1, axis=1)
    desc2 = np.unpackbits(desc2, axis=1)
    #distances = distance.cdist(desc1, desc2, metric=hamming_distance)
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



def mnn_matcher_L2(desc1, desc2, constant_cut):
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

    if 'orb' in args_opt:
        s_desc1 = s_desc1.astype(np.uint8)
        s_desc2 = s_desc2.astype(np.uint8)
        distance_eval = cv2.NORM_HAMMING
        distance_eval_FLANN = 6
    else:
        distance_eval = cv2.NORM_L2
        distance_eval_FLANN = 1
    

    if match == 'BF':
        bf = cv2.BFMatcher(distance_eval, True)
        matches = bf.match(s_desc1, s_desc2)
    elif match == 'BF_KNN':
        if args_opt == "superpoint":
            constant = 0.95
        elif args_opt == "orb":
            constant = 0.9
        else:
            constant = 0.75
        matches = bfknn_matcher(s_desc1, s_desc2, distance_eval, constant)
    elif match == 'FLANN_KNN':
        matches = flannknn_matcher(s_desc1, s_desc2, distance_eval_FLANN, constant)
    elif match == 'MNN':
        if 'orb' in args_opt:
            matches = mnn_matcher_hamming(desc1, desc2, constant)
        else:
            matches = mnn_matcher_L2(s_desc1, s_desc2, constant)
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
    descriptor_configs = {
        'Proposed': ('superpoint', image_mode, 512),
        'sphorb': ('sphorb', image_mode, 640),
        'sift': ('sift', image_mode, 512),
        'orb': ('orb', image_mode, 512),
        'spoint': ('superpoint', image_mode, 512),
        'akaze': ('akaze', image_mode, 512),
        'alike': ('alike', image_mode, 512),
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