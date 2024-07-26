import copy
import os
import cv2
import glob
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch
import utils.superpoint.train_sp.superpoint as train_sp
from utils.ALIKE.alike import ALike, configs
from utils.ALIKED.nets.aliked import ALIKED
from utils.coord    import coord_3d
from utils.ransac   import *
from utils.keypoint import *
from utils.metrics  import *
from utils.method  import *
from utils.camera_recovering import *
from utils.matching import *
from utils.spherical_module import *
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dim = [1024, 512]
Y_remap, X_remap, make_map_time = make_image_map([512, 1024])


def main(args):
    descriptor = args.descriptor
    proposed_method_flag = False
    if descriptor[0] == 'p':
        descriptor = descriptor[1:]
        proposed_method_flag = True
    model = get_model(descriptor)
    logging.basicConfig(level=logging.INFO) 
    logging.info("Press 'q' to stop!")
    cv2.namedWindow(descriptor)   

    image_loader = ImageLoader(args.input)
    tracker = SimpleTracker()
    runtime_list = []
    Trueratio_list = []
    Nmatches_list = []
    Matchingscore_list = []
    progress_bar = tqdm(image_loader)
    for img in progress_bar:
        if img is None:
            break
        if proposed_method_flag:
            kpts, desc, time_fp = keypoint_proposed(img, descriptor, model)
        else:
            kpts, desc, time_fp = get_kptsdesc(img, descriptor, model)
            kpts, desc, _ = sort_key_div_np(kpts, desc.T, 1000)
        
        out, x_eq1, x_eq2 = tracker.update(img, kpts, desc, descriptor)
        Nmatches = len(x_eq1)
        inlier_idx = []
        status = ""
        if Nmatches > 8:
            runtime_list.append(time_fp)
            x_3d1,x_3d2 = coord_3d(x_eq1, dim), coord_3d(x_eq2, dim)
            E, cam, inlier_idx = get_cam_pose_by_ransac(x_3d1.copy().T,x_3d2.copy().T, get_E = True, solver="SK")
            Nmatches_list.append(Nmatches)
            Trueratio_list.append(sum(inlier_idx) / len(kpts))
            Matchingscore_list.append(sum(inlier_idx) / Nmatches)
            ave_fps = (1. / np.stack(runtime_list)).mean()
            status = f"Fps:{ave_fps:.1f}, Keypoints/Matches/Truematches: {len(kpts)}/{Nmatches}/{sum(inlier_idx)}"

            progress_bar.set_description(status)
            cv2.setWindowTitle(descriptor, descriptor + ': ' + status)
            cv2.imshow(descriptor, out)
            if cv2.waitKey(1) == ord('q'):
                break
    
    print(f"Average; FPS: {ave_fps:.1f}, Matches: {np.mean(Nmatches_list):.1f}", 
          f"True Ratio: {np.mean(Trueratio_list):.3f}, Matching Score: {np.mean(Matchingscore_list):.3f}")
    logging.info('Finished!')
    logging.info('Press any key to exit!')
    cv2.waitKey()


class ImageLoader(object):
    def __init__(self, filepath: str):
        self.N = 3000
        if filepath.startswith('camera'):
            camera = int(filepath[6:])
            self.cap = cv2.VideoCapture(camera)
            if not self.cap.isOpened():
                raise IOError(f"Can't open camera {camera}!")
            logging.info(f'Opened camera {camera}')
            self.mode = 'camera'
        elif os.path.exists(filepath):
            if os.path.isfile(filepath):
                self.cap = cv2.VideoCapture(filepath)
                if not self.cap.isOpened():
                    raise IOError(f"Can't open video {filepath}!")
                rate = self.cap.get(cv2.CAP_PROP_FPS)
                self.N = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
                duration = self.N / rate
                logging.info(f'Opened video {filepath}')
                logging.info(f'Frames: {self.N}, FPS: {rate}, Duration: {duration}s')
                self.mode = 'video'
            else:
                self.images = glob.glob(os.path.join(filepath, '*.png')) + \
                              glob.glob(os.path.join(filepath, '*.jpg')) + \
                              glob.glob(os.path.join(filepath, '*.ppm'))
                self.images.sort()
                self.N = len(self.images)
                logging.info(f'Loading {self.N} images')
                self.mode = 'images'
        else:
            raise IOError('Error filepath (camerax/path of images/path of videos): ', filepath)

    def __getitem__(self, item):
        if self.mode == 'camera' or self.mode == 'video':
            if item > self.N:
                return None
            ret, img = self.cap.read()
            if not ret:
                raise "Can't read image from camera"
            if self.mode == 'video':
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, item)
        elif self.mode == 'images':
            filename = self.images[item]
            img = cv2.imread(filename)
            if img is None:
                raise Exception('Error reading image %s' % filename)        
        return img

    def __len__(self):
        return self.N


class SimpleTracker(object):
    def __init__(self):
        self.pts_prev = None
        self.desc_prev = None

    def update(self, img, pts, desc, descriptor):
        mpts1, mpts2 = [], []
        if self.pts_prev is None:
            self.pts_prev = pts
            self.desc_prev = desc

            out = copy.deepcopy(img)
            for pt1 in pts:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                cv2.circle(out, p1, 1, (0, 0, 255), -1, lineType=16)
        else:
            matches = self.matcher_parser(self.desc_prev, desc, descriptor)
            mpts1, mpts2 = self.pts_prev[matches[:, 0]], pts[matches[:, 1]]

            out = copy.deepcopy(img)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, (0, 255, 0), lineType=16)
                cv2.circle(out, p2, 1, (0, 0, 255), -1, lineType=16)

            self.pts_prev = pts
            self.desc_prev = desc

        return out, mpts1, mpts2


    def matcher_parser(self, desc1, desc2, descriptor):
        if descriptor in ['alike', 'aliked']:
            return self.alike_mnn_matcher(desc1, desc2)
        elif descriptor in ['orb', 'akaze']:
            return self.mnn_matcher_hamming(desc1, desc2)
        elif descriptor in ['sift', 'spoint']:
            return self.mnn_matcher_L2(desc1, desc2)


    def alike_mnn_matcher(self, desc1, desc2):
        sim = desc1 @ desc2.transpose()
        sim[sim < 0.2] = 0
        nn12 = np.argmax(sim, axis=1)
        nn21 = np.argmax(sim, axis=0)
        ids1 = np.arange(0, sim.shape[0])
        mask = (ids1 == nn21[nn12])
        matches = np.stack([ids1[mask], nn12[mask]])
        return matches.transpose()


    def mnn_matcher_hamming(self, desc1, desc2):
        desc1 = desc1.astype(np.uint8)
        desc2 = desc2.astype(np.uint8)
        desc1 = np.unpackbits(desc1, axis=1)
        desc2 = np.unpackbits(desc2, axis=1)
        distances = self.hamming_distance_optimized(desc1, desc2)
        distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        distances[distances > 0.3] = np.inf

        nn12 = np.argmin(distances, axis=1)
        nn21 = np.argmin(distances, axis=0)
        ids1 = np.arange(desc1.shape[0])
        mask = (ids1 == nn21[nn12])

        matches = np.stack([ids1[mask], nn12[mask]], axis=1)

        return matches


    def hamming_distance_optimized(self, desc1, desc2):
        return np.bitwise_xor(desc1[:, None, :], desc2).sum(axis=-1)
    

    def mnn_matcher_L2(self, desc1, desc2):
        d1_square = np.sum(np.square(desc1), axis=1, keepdims=True)
        d2_square = np.sum(np.square(desc2), axis=1, keepdims=True)
        distances = np.sqrt(d1_square - 2 * np.dot(desc1, desc2.T) + d2_square.T)
        distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        distances[distances > 0.3] = np.inf

        nn12 = np.argmin(distances, axis=1)
        nn21 = np.argmin(distances, axis=0)
        ids1 = np.arange(0, distances.shape[0])
        mask = (ids1 == nn21[nn12])
        matches = np.stack([ids1[mask], nn12[mask]])
        
        return matches.transpose()


def get_kptsdesc(img, descriptor, model):
    if descriptor == 'alike':
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        time_feature1 = time.perf_counter()
        pred = model(img_rgb)
        time_feature2 = time.perf_counter()
        kpts = pred['keypoints']
        desc = pred['descriptors']
        scores = pred['scores']
        kpts = np.column_stack((kpts, scores))
    elif descriptor == 'aliked':
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        time_feature1 = time.perf_counter()
        pred = model.run(img_rgb)
        time_feature2 = time.perf_counter()
        kpts = pred['keypoints']
        desc = pred['descriptors']
        scores = pred['scores']
        kpts = np.column_stack((kpts, scores))
    elif descriptor == 'orb':
        time_feature1 = time.perf_counter()
        kpts, desc = model.detectAndCompute(img, None)
        time_feature2 = time.perf_counter()
        kpts = np.array([(kp.pt[0], kp.pt[1], kp.response) for kp in kpts])
        desc = np.array(desc)
    elif descriptor == 'sift':
        time_feature1 = time.perf_counter()
        kpts, desc = model.detectAndCompute(img, None)
        time_feature2 = time.perf_counter()
        kpts = np.array([(kp.pt[0], kp.pt[1], kp.response) for kp in kpts])
        desc = np.array(desc)
    elif descriptor == 'akaze':
        time_feature1 = time.perf_counter()
        kpts, desc = model.detectAndCompute(img, None)
        time_feature2 = time.perf_counter()
        kpts = np.array([(kp.pt[0], kp.pt[1], kp.response) for kp in kpts])
        desc = np.array(desc)
    elif descriptor == 'spoint':
        img = process_img(img)
        time_feature1 = time.perf_counter()
        pts, desc, _ = sp_model.run(img)
        time_feature2 = time.perf_counter()
        kpts = np.zeros((pts.shape[1],4))
        kpts[:,0] = pts[0,:]
        kpts[:,1] = pts[1,:]
        kpts[:,2] = pts[2,:]
        kpts[:,3] = pts[2,:]
        desc = np.array(desc.T)

    return kpts, desc, time_feature2 - time_feature1


def keypoint_proposed(img, descriptor, model, padding_length=34):
    img_hw = img.shape[:2]
    img_hw_crop = (img_hw[0]//2+padding_length*2, img_hw[1]*3//4+padding_length*2)
    crop_start_xy = ((img_hw[0]-img_hw_crop[0])//2 - 1, (img_hw[1]-img_hw_crop[1])//2 - 1)
    img1, img2, remap_time = remap_crop_image(img, (Y_remap, X_remap), img_hw_crop, crop_start_xy)

    pts1_, desc1_, feature_time1 = get_kptsdesc(img1, descriptor, model)
    pts2_, desc2_, feature_time2 = get_kptsdesc(img2, descriptor, model)
    feature_time = feature_time1 + feature_time2 + remap_time

    pts1_ = torch.tensor(pts1_).float()
    pts2_ = torch.tensor(pts2_).float()
    desc1_ = torch.tensor(desc1_.T).float()
    desc2_ = torch.tensor(desc2_.T).float()


    pts1_ = add_offset_to_image(pts1_, crop_start_xy)
    pts2_ = add_offset_to_image(pts2_, crop_start_xy)
    pts2_ = convert_coordinates_vectorized(pts2_, img_hw)
    pts1_, desc1_ = filter_keypoints(pts1_, desc1_, img_hw)
    pts2_, desc2_ = filter_keypoints(pts2_, desc2_, img_hw, invert_mask=True)
    image_kp = torch.cat((pts1_, pts2_), dim=0)
    image_desc = torch.cat((desc1_, desc2_), dim=1)
    image_kp, image_desc, _ = sort_key_div_np(image_kp.numpy(), image_desc.numpy(), 1000)

    return image_kp, image_desc, feature_time


def get_model(descriptor):
    feature_limit = 1000
    if descriptor == 'orb':
        model = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE, nfeatures=feature_limit)
    elif descriptor == 'sift':
        model = cv2.SIFT_create(nfeatures=feature_limit)
    elif descriptor == 'akaze':
        model = cv2.AKAZE_create()
    elif descriptor == 'spoint':
        model = train_sp.SuperPointFrontend(weights_path = 'utils/models/superpoint-trained.pth.tar', 
                                     nms_dist= 4,
                                     conf_thresh = 0.015,
                                     nn_thresh= 0.7,
                                     cuda = True)
    elif descriptor == 'alike':
        model = ALike(**configs["alike-n"],
                        device=device,
                        top_k=-1,
                        scores_th=0.2,
                        n_limit=feature_limit
                        )
    elif descriptor == 'aliked':
        model = ALIKED(model_name="aliked-n16", # aliked-t16, aliked-n32,
                          device="cuda",
                          top_k=-1,
                          scores_th=0.2,
                          n_limit=feature_limit
                          )
    return model


def process_img(img):
    H, W = img.shape[0], img.shape[1]
    grayim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayim = (grayim.astype('float32') / 255.)
    return grayim


def sort_key_div_np(pts1, desc1, points):
    ind1 = np.argsort(pts1[:,2])[::-1]
    max1 = min(points, len(ind1))
    ind1 = ind1[:max1]
    pts1 = pts1[ind1]
    scores1 = pts1[:, 2:3].copy()
    desc1 = desc1[:, ind1]
    pts1 = np.concatenate((pts1[:,:2], np.ones((len(pts1), 1))), axis=1)
    desc1 = desc1.T

    return pts1, desc1, scores1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ALike Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory or movie file or "camera0" (for webcam0).')
    parser.add_argument('--descriptor',  default='orb')
    args = parser.parse_args()

    main(args)