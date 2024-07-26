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
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
orb_model = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE, nfeatures=1000)
sift_model = cv2.SIFT_create(nfeatures=1000)
akaze_model = cv2.AKAZE_create()
sp_model = train_sp.SuperPointFrontend(weights_path = 'utils/models/superpoint-trained.pth.tar', 
                                 nms_dist= 4,
                                 conf_thresh = 0.015,
                                 nn_thresh= 0.7,
                                 cuda = True)
alike_model = ALike(**configs["alike-n"],
                    device=device,
                    top_k=-1,
                    scores_th=0.2,
                    n_limit=1000
                    )
aliked_model = ALIKED(model_name="aliked-n16", # aliked-t16, aliked-n32,
                      device="cuda",
                      top_k=-1,
                      scores_th=0.2,
                      n_limit=1000
                      )


def main(args):
    descriptor = args.descriptor
    logging.basicConfig(level=logging.INFO) 
    logging.info("Press 'q' to stop!")
    cv2.namedWindow("descriptor")   


    image_loader = ImageLoader(args.input)
    tracker = SimpleTracker()
    runtime_list = []
    Trueratio_list = []
    Nmatches_list = []
    progress_bar = tqdm(image_loader)
    for img in progress_bar:
        if img is None:
            break
        kpts, desc, time_fp = get_kptsdesc(img, descriptor)
        runtime_list.append(time_fp)
        out, Nmatches = tracker.update(img, kpts, desc, descriptor)
        Nmatches_list.append(Nmatches)
        Trueratio_list.append(Nmatches / len(kpts))
        
        ave_fps = (1. / np.stack(runtime_list)).mean()
        status = f"Fps:{ave_fps:.1f}, Keypoints/Matches: {len(kpts)}/{Nmatches}"
        progress_bar.set_description(status)

        cv2.setWindowTitle(descriptor, descriptor + ': ' + status)
        cv2.imshow(descriptor, out)
        if cv2.waitKey(1) == ord('q'):
            break
    
    print(f"Average; FPS: {ave_fps:.1f}, Matches: {np.mean(Nmatches_list):.1f}", 
          f"True Ratio: {np.mean(Trueratio_list):.3f}")
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
        N_matches = 0
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
            N_matches = len(matches)

            out = copy.deepcopy(img)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(out, p1, p2, (0, 255, 0), lineType=16)
                cv2.circle(out, p2, 1, (0, 0, 255), -1, lineType=16)

            self.pts_prev = pts
            self.desc_prev = desc

        return out, N_matches


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


def get_kptsdesc(img, descriptor):
    if descriptor == 'alike':
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        time_feature1 = time.perf_counter()
        pred = alike_model(img_rgb)
        time_feature2 = time.perf_counter()
        kpts = pred['keypoints']
        desc = pred['descriptors']
    elif descriptor == 'aliked':
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        time_feature1 = time.perf_counter()
        pred = aliked_model.run(img_rgb)
        time_feature2 = time.perf_counter()
        kpts = pred['keypoints']
        desc = pred['descriptors']
    elif descriptor == 'orb':
        time_feature1 = time.perf_counter()
        kpts, desc = orb_model.detectAndCompute(img, None)
        time_feature2 = time.perf_counter()
        kpts = np.array([kp.pt for kp in kpts])
        desc = np.array(desc)
    elif descriptor == 'sift':
        time_feature1 = time.perf_counter()
        kpts, desc = sift_model.detectAndCompute(img, None)
        time_feature2 = time.perf_counter()
        kpts = np.array([kp.pt for kp in kpts])
        desc = np.array(desc)
    elif descriptor == 'akaze':
        time_feature1 = time.perf_counter()
        kpts, desc = akaze_model.detectAndCompute(img, None)
        time_feature2 = time.perf_counter()
        kpts = np.array([kp.pt for kp in kpts])
        desc = np.array(desc)

    return kpts, desc, time_feature2 - time_feature1


def keypoint_proposed(img, opt, scale_factor, img_hw, padding_length=50):
    Y_remap, X_remap, make_map_time = make_image_map(img_hw)
    img_hw_crop = (img_hw[0]//2+padding_length*2, img_hw[1]*3//4+padding_length*2)
    crop_start_xy = ((img_hw[0]-img_hw_crop[0])//2 - 1, (img_hw[1]-img_hw_crop[1])//2 - 1)

    img1, img2, remap_time = remap_crop_image(img, (Y_remap, X_remap), img_hw_crop, crop_start_xy)
    img1 = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0)

    img1 = F.interpolate(img1, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)
    img2 = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0)
    img2 = F.interpolate(img2, scale_factor=scale_factor, mode='bilinear', align_corners=False, recompute_scale_factor=True).squeeze(0)

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

    return image_kp, image_desc, make_map_time, remap_time, feature_time


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ALike Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory or movie file or "camera0" (for webcam0).')
    parser.add_argument('--model', choices=['alike-t', 'alike-s', 'alike-n', 'alike-l', "test"], default="alike-n",
                        help="The model configuration")
    parser.add_argument('--descriptor', choices=['alike', 'orb', 'sift', 'aliked', 'akaze'], default='alike')
    parser.add_argument('--n_limit', type=int, default=1000,
                        help='Maximum number of keypoints to be detected (default: 5000).')
    args = parser.parse_args()

    main(args)