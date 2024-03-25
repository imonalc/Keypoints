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
sys.path.append('..')
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from SuperGluePretrainedNetwork.models.superglue import SuperGlue

from LightGlue.lightglue import LightGlue

sys.path.append('..'+'/SphereGlue')
from SphereGlue.model.sphereglue import SphereGlue



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



def bfknn_matcher(s_desc1, s_desc2, distance_eval, constant):
    thresh = 0.95
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



def hamming_distance(x, y):
    return np.sum(x != y, axis=-1)

def mnn_matcher_hamming(desc1, desc2, constant_cut):
    desc1 = np.unpackbits(desc1, axis=1)
    desc2 = np.unpackbits(desc2, axis=1)
    distances = distance.cdist(desc1, desc2, metric=hamming_distance)


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





class Matching(torch.nn.Module):
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
            #print(pred)
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        matching_b = time.perf_counter()
        # Batch all features
        # We should either have i) one image per batch, or
        # ii) the same number of local features for all images in the batch.
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        pred = {**pred, **self.superglue(data)}
        matching_a = time.perf_counter()

        return pred, matching_b, matching_a


def superpoint_superglue(img_o, img_r, device):
    config = {
        'superpoint': {
            'max_keypoints': 1000
        },
        'superglue': {
            'weights': "indoor",
            'match_threshold': 0.03
        }
    }
    x1_, x2_ = [], []
    matching = Matching(config).eval().to(device)
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
    
    pred, t_matching_b, t_matching_a = matching(data)
  
    kpt1 = pred["keypoints0"][0].cpu().numpy()
    kpt1 = np.hstack((kpt1, np.ones((kpt1.shape[0], 1))))
    kpt2 = pred["keypoints1"][0].cpu().numpy()
    kpt2 = np.hstack((kpt2, np.ones((kpt2.shape[0], 1))))
    matches1 = pred["matches0"][0].cpu().numpy()
    matches2 = pred["matches1"][0].cpu().numpy()
    for i in range(len(matches1)):
        if matches1[i] == -1: continue
        x1_.append(kpt1[i])
        x2_.append(kpt2[matches1[i]])
    s_pts1, s_pts2 = np.array(kpt1), np.array(kpt2)
    x1_, x2_ = np.array(x1_), np.array(x2_)

    return s_pts1, s_pts2, x1_, x2_, t_matching_b, t_matching_a


def superglue_matching(pts1, pts2, desc1, desc2, scores1, scores2, img_o, img_r, device):
    config = {
        'superglue': {
            #'weights': "indoor",
            #'match_threshold': 0.03
        }
    }
    x1_, x2_ = [], []
    matching = Matching(config).eval().to(device)
    img_o_tensor = torch.from_numpy(img_o).float()/255.0
    img_o_tensor = img_o_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    img_r_tensor = torch.from_numpy(img_r).float()/255.0
    img_r_tensor = img_r_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    img_o_tensor = img_o_tensor[:, 0:1, :, :]
    img_r_tensor = img_r_tensor[:, 0:1, :, :]
    
    
    keypoints0 = np.delete(pts1, 2, axis=1)
    keypoints0_tensor = torch.tensor(keypoints0,dtype=torch.float).to(device)
    keypoints1 = np.delete(pts2, 2, axis=1)
    keypoints1_tensor = torch.tensor(keypoints1,dtype=torch.float).to(device)
    
    descriptors0_tensor = torch.tensor(desc1.T,dtype=torch.float).to(device)
    descriptors1_tensor = torch.tensor(desc2.T,dtype=torch.float).to(device)
    
    scores0_tensor = scores1.squeeze(1).to(device)
    scores1_tensor = scores2.squeeze(1).to(device)
    
    data = {
        "keypoints0": [keypoints0_tensor],
        "descriptors0": [descriptors0_tensor],
        "scores0": (scores0_tensor, ),
        "keypoints1": [keypoints1_tensor],
        "descriptors1": [descriptors1_tensor],
        "scores1": (scores1_tensor, ),
        "image0":img_o_tensor,
        "image1":img_r_tensor,
    }
    
    pred, t_matching_b, t_matching_a = matching(data)
    kpt1 = keypoints0_tensor.cpu().numpy()
    kpt1 = np.hstack((kpt1, np.ones((kpt1.shape[0], 1))))
    kpt2 = keypoints1_tensor.cpu().numpy()
    kpt2 = np.hstack((kpt2, np.ones((kpt2.shape[0], 1))))
    matches1 = pred[0]["matches0"][0].cpu().numpy()
    matches2 = pred[0]["matches1"][0].cpu().numpy()
    for i in range(len(matches1)):
        if matches1[i] == -1: continue
        x1_.append(kpt1[i])
        x2_.append(kpt2[matches1[i]])
    s_pts1, s_pts2 = np.array(kpt1), np.array(kpt2)
    x1_, x2_ = np.array(x1_), np.array(x2_)

    return s_pts1, s_pts2, x1_, x2_, t_matching_b, t_matching_a


def lightglue_matching(pts1, pts2, desc1, desc2, scores1, scores2, device):
    matcher = LightGlue(features="superpoint").eval().to(device)
    x1_, x2_ = [], []
    keypoints0 = np.delete(pts1, 2, axis=1)
    keypoints0_tensor = torch.tensor(keypoints0,dtype=torch.float).unsqueeze(0).to('cuda:0')
    keypoints1 = np.delete(pts2, 2, axis=1)
    keypoints1_tensor = torch.tensor(keypoints1,dtype=torch.float).unsqueeze(0).to('cuda:0')
    descriptors0_tensor = torch.tensor(desc1,dtype=torch.float).unsqueeze(0).to('cuda:0')
    descriptors1_tensor = torch.tensor(desc2,dtype=torch.float).unsqueeze(0).to('cuda:0')
    scores0_tensor = scores1.squeeze().to('cuda:0')
    scores1_tensor = scores2.squeeze().to('cuda:0')
    data = {
        "image0":{
            "keypoints": keypoints0_tensor,
            "keypoint_scores": scores0_tensor,
            "descriptors": descriptors0_tensor
        },
        "image1":{
            "keypoints": keypoints1_tensor,
            "keypoint_scores": scores1_tensor,
            "descriptors": descriptors1_tensor
        }
    }

    t_matching_b = time.perf_counter()
    pred = matcher(data)
    matches = pred['matches']
    
    matches1 = keypoints0[matches[0][:, 0].cpu().numpy(), :] 
    matches2 = keypoints1[matches[0][:, 1].cpu().numpy(), :] 
    for i in range(len(matches1)):
        x1_.append(matches1[i])
        x2_.append(matches2[i])
    t_matching_a = time.perf_counter()

    s_pts1, s_pts2 = np.array(keypoints0), np.array(keypoints1)
    x1_, x2_ = np.array(x1_), np.array(x2_)

    return s_pts1, s_pts2, x1_, x2_, t_matching_b, t_matching_a



def sphereglue_matching(pts1, pts2, desc1, desc2, scores1, scores2, points, device):
    config = {'K': 2, 
              'GNN_layers': ['cross'], 
              'match_threshold': 0.2, 
              'sinkhorn_iterations': 20, 
              'aggr': 'add', 
              'knn': 20, 
              'max_kpts': points,
              'descriptor_dim': 256, 
              'output_dim': 512
    }
    matching = SphereGlue(config).to(device)
    model_path = '../SphereGlue/model_weights/superpoint/autosaved.pt'
    ckpt_data = torch.load(model_path)
    matching.load_state_dict(ckpt_data["MODEL_STATE_DICT"])
    matching.eval()
    x1_, x2_ = [], []
    
    
    keypoints0 = np.delete(pts1, 2, axis=1)
    keypoints0_c = convert_to_spherical_coordinates(keypoints0, 1024, 512)
    keypoints0_tensor = torch.tensor(keypoints0_c,dtype=torch.float).unsqueeze(0).to(device)
    keypoints1 = np.delete(pts2, 2, axis=1)
    keypoints1_c = convert_to_spherical_coordinates(keypoints1, 1024, 512)
    keypoints1_tensor = torch.tensor(keypoints1_c,dtype=torch.float).unsqueeze(0).to(device)
    
    descriptors0_tensor = torch.tensor(desc1,dtype=torch.float).unsqueeze(0).to(device)
    descriptors1_tensor = torch.tensor(desc2,dtype=torch.float).unsqueeze(0).to(device)
    
    scores0_tensor = scores1.squeeze(1).unsqueeze(0).to(device)
    scores1_tensor = scores2.squeeze(1).unsqueeze(0).to(device)
    
    data = {
        "unitCartesian1": keypoints0_tensor,
        "h1": descriptors0_tensor,
        "scores1": scores0_tensor,
        "unitCartesian2": keypoints1_tensor,
        "h2": descriptors1_tensor,
        "scores2": scores1_tensor,
    }

    t_matching_b = time.perf_counter()
    pred = matching(data)
    t_matching_a = time.perf_counter()

    kpt1 = keypoints0.copy()
    kpt1 = np.hstack((kpt1, np.ones((kpt1.shape[0], 1))))
    kpt2 = keypoints1.copy()
    kpt2 = np.hstack((kpt2, np.ones((kpt2.shape[0], 1))))
    matches1 = pred["matches0"].cpu().numpy()
    
    for i in range(len(matches1[0])):
        if matches1[0][i] == -1: continue
        x1_.append(kpt1[i])
        x2_.append(kpt2[matches1[0][i]])
    s_pts1, s_pts2 = np.array(kpt1), np.array(kpt2)
    x1_, x2_ = np.array(x1_), np.array(x2_)

    return s_pts1, s_pts2, x1_, x2_, t_matching_b, t_matching_a



def convert_to_spherical_coordinates(keypoints, image_width, image_height, device="cuda"):
    longitude = (keypoints[:, 0] / image_width) * 360 - 180
    latitude = (keypoints[:, 1] / image_height) * 180 - 90

    lon_rad = np.deg2rad(longitude)
    lat_rad = np.deg2rad(latitude)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    keypoints3D = np.stack([x, y, z], axis=-1)
    keypoints3D_tensor = torch.tensor(keypoints3D, dtype=torch.float).to(device)

    return keypoints3D_tensor



### legacy

def get_descriptor2(descriptor):
    descriptor_configs = {
        'sphorb': ('sphorb', 'erp', 640, 0),
        'sift': ('sift', 'erp', 512, 0),
        'tsift': ('sift', 'tangent', 512, 0),
        'orb': ('orb', 'erp', 512, 0),
        'torb': ('orb', 'tangent', 512, 0),
        'spoint': ('superpoint', 'erp', 512, 0),
        'tspoint': ('superpoint', 'tangent', 512, 0),
        'akaze': ('akaze', 'erp', 512, 0),
        'takaze': ('akaze', 'tangent', 512, 0),
        'alike': ('alike', 'erp', 512, 0),
        'talike': ('alike', 'tangent', 512, 0),
        'Proposed1': ('superpoint', 'erp', 512, 1),
        'Proposed01': ('superpoint', 'erp', 512, 2),
        'Proposed05': ('superpoint', 'erp', 512, 3),
        'Proposed2': ('superpoint', 'erp', 512, 8),
        'Proposed3': ('superpoint', 'erp', 512, 4),
        'Proposed4': ('superpoint', 'erp', 512, 9),
        'Proposed5': ('superpoint', 'erp', 512, 5),
        'Proposed10': ('superpoint', 'erp', 512, 6),
        'Proposed_nolimit': ('superpoint', 'erp', 512, 10000),
        'Proposed_nolimit2': ('superpoint', 'tangent', 512, 7),
        'Ltspoint': ('superpoint', 'tangent', 512, 10),
        'Ftspoint': ('superpoint', 'tangent', 512, 11),
        'superglue': ('superpoint', 'tangent', 512, 100),
        'lightglue': ('superpoint', 'tangent', 512, 101),
        'sphereglue': ('superpoint', 'tangent', 512, 102),
        'Proposed2_001': ('superpoint', 'erp', 512, 1000),
        'Proposed2_003': ('superpoint', 'erp', 512, 1001),
        'Proposed2_005': ('superpoint', 'erp', 512, 1002),
        'Proposed2_01': ('superpoint', 'erp', 512, 1003),
        'Proposed2_02': ('superpoint', 'erp', 512, 1004),
        'Proposed3_1': ('superpoint', 'tangent', 512, 10001),
        'Proposed3_2': ('superpoint', 'tangent', 512, 10002),
        'Proposed3_3': ('superpoint', 'tangent', 512, 10003),
        'Proposed3_4': ('superpoint', 'tangent', 512, 10004)
    }

    return descriptor_configs.get(descriptor, ('unknown', 'unknown', 0, -1))



def matched_points2(pts1, pts2, desc1, desc2, opt, args_opt, match='ratio', use_new_method=0):
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
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, True) ##orb
        #bf = cv2.BFMatcher(cv2.NORM_L2, True)
        matches = bf.match(s_desc1, s_desc2)
    elif use_new_method in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        matches_idx = mnn_matcher(s_desc1, s_desc2, use_new_method)
        matches = [cv2.DMatch(i, j, 0) for i, j in matches_idx]
    elif use_new_method in [1000, 1001, 1002, 1003, 1004, 10001, 10002, 10003, 10004, 10000]:
        matches_idx = mnn_matcher_new(s_desc1, s_desc2, use_new_method)
        matches = [cv2.DMatch(i, j, 0) for i, j in matches_idx]
    elif use_new_method == 10:
        thresh = 0.75
        bf = cv2.BFMatcher(cv2.NORM_L2, False)
        matches1 = bf.knnMatch(s_desc1,s_desc2, k=2)
        matches2 = bf.knnMatch(s_desc2,s_desc1, k=2)
        good1 = []
        for m, n in matches1:
            if m.distance < thresh * n.distance:
                good1.append(m)
        good2 = []
        for m, n in matches2:
            if m.distance < thresh * n.distance:
                good2.append(m)
        good = []
        for m1 in good1:
            for m2 in good2:
                if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
                    good.append(m1)
                    break
        matches = good
    elif use_new_method == 11:
        thresh = 0.75
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(s_desc1, s_desc2, k=2)
        good = []
        for m,n in matches:
            if m.distance < thresh * n.distance:
                good.append(m)
        matches = good
    elif match == 'ratio':
        thresh = 0.75
        bf = cv2.BFMatcher(cv2.NORM_L2, False)
        matches = bf.knnMatch(s_desc1,s_desc2, k=2)
        good = []
        for m,n in matches:
            if m.distance < thresh * n.distance:
                good.append(m)
        matches = good
    else:
        raise ValueError("Invalid matching method specified.")
    


    M = np.zeros((2,len(matches)))
    for ind, match in zip(np.arange(len(matches)),matches):
        M[0,ind] = match.queryIdx
        M[1,ind] = match.trainIdx


    return s_pts1, s_pts2, s_pts1[M[0,:].astype(int),:3], s_pts2[M[1,:].astype(int),:3]



def mnn_matcher_old(desc1, desc2, use_new_method):
    sim = desc1 @ desc2.transpose()
    sim = (sim - np.min(sim))/ (np.max(sim) - np.min(sim))
    if use_new_method == 1:
        dec = 1
    elif use_new_method == 2:
        dec = 0.1
    elif use_new_method == 3:
        dec = 0.5
    elif use_new_method == 4:
        dec = 3
    elif use_new_method == 5:
        dec = 5
    elif use_new_method == 6:
        dec = 10
    elif use_new_method == 7:
        dec = 100
    threshold = np.percentile(sim, 100-dec)
    
    sim[sim < threshold] = 0
    nn12 = np.argmax(sim, axis=1)
    nn21 = np.argmax(sim, axis=0)
    ids1 = np.arange(0, sim.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]])
    return matches.transpose()


def mnn_matcher(desc1, desc2, use_new_method):
    d1_square = np.sum(np.square(desc1), axis=1, keepdims=True)
    d2_square = np.sum(np.square(desc2), axis=1, keepdims=True)
    distances = np.sqrt(d1_square - 2 * np.dot(desc1, desc2.T) + d2_square.T)
    distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

    if use_new_method == 1:
        dec = 1
    elif use_new_method == 2:
        dec = 0.1
    elif use_new_method == 3:
        dec = 0.5
    elif use_new_method == 4:
        dec = 3
    elif use_new_method == 5:
        dec = 5
    elif use_new_method == 6:
        dec = 10
    elif use_new_method == 7:
        dec = 100
    elif use_new_method == 8:
        dec = 2
    elif use_new_method == 9:
        dec = 4
    threshold = np.percentile(distances, dec) 
    distances[distances > threshold] = np.max(distances)
    #print("border:", threshold)

    nn12 = np.argmin(distances, axis=1)
    nn21 = np.argmin(distances, axis=0)
    ids1 = np.arange(0, distances.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.stack([ids1[mask], nn12[mask]])

    return matches.transpose()


def mnn_matcher_new(desc1, desc2, use_new_method):
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
    if use_new_method == 1000:
        dec = 0.01
    if use_new_method == 1001:
        dec = 0.03
    if use_new_method == 1002:
        dec = 0.05
    if use_new_method == 1003:
        dec = 0.1
    if use_new_method == 1004:
        dec = 0.2
    if use_new_method == 10001:
        dec = 0.1
    if use_new_method == 10002:
        dec = 0.05
    if use_new_method == 10003:
        dec = 0.2
    if use_new_method == 10004:
        dec = 0.3
    if use_new_method == 10000:
        return sorted_matches_with_distances[:, :2].transpose().astype(int).transpose()
    num_to_remove = int(len(sorted_matches_with_distances) * dec)
    matches = sorted_matches_with_distances[:-num_to_remove, :2].transpose().astype(int)
    #print("border:", sorted_matches_with_distances[-num_to_remove-1:-num_to_remove, :])

    return matches.transpose()