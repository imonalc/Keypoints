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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

orb_model = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE, nfeatures=1000)
sift_model = cv2.SIFT_create(nfeatures=1000)
akaze_model = cv2.AKAZE_create()
sp_model = train_sp.SuperPointFrontend(weights_path = 'utils/models/superpoint-trained.pth.tar', 
                                 nms_dist= 4, #nms_dist, 
                                 conf_thresh = 0.015, #conf_thresh, 
                                 nn_thresh= 0.7, #nn_thresh, 
                                 cuda = True ) # 4 0.015 or 3 0.005
alike_model = ALike(**configs["alike-n"],
                    device=device,
                    top_k=-1,
                    scores_th=0.2,
                    n_limit=1000
                    )

aliked_model = ALIKED(model_name="aliked-n16",
                      device="cuda",
                      top_k=-1,
                      scores_th=0.2,
                      n_limit=1000
                      )


def main(args):
    descriptor = args.descriptor
    logging.basicConfig(level=logging.INFO)    
    image_loader = ImageLoader(args.input)
    tracker = SimpleTracker()
    
    if not args.no_display:
        logging.info("Press 'q' to stop!")
        cv2.namedWindow(args.model)

    
    runtime = []
    progress_bar = tqdm(image_loader)
    for img in progress_bar:
        if img is None:
            break
        if descriptor == 'alike':
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = alike_model(img_rgb, sub_pixel=not args.no_sub_pixel)
            kpts = pred['keypoints']
            desc = pred['descriptors']
        elif descriptor == 'aliked':
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pred = aliked_model.run(img_rgb)
            kpts = pred['keypoints']
            desc = pred['descriptors']
        elif descriptor == 'orb':
            orb = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE, nfeatures=1000)
            kpts, desc = orb.detectAndCompute(img, None)
            kpts = np.array([kp.pt for kp in kpts])
            desc = np.array(desc)
        elif descriptor == 'sift':
            sift = cv2.SIFT_create(nfeatures=1000)
            kpts, desc = sift.detectAndCompute(img, None)
            kpts = np.array([kp.pt for kp in kpts])
            desc = np.array(desc)
        elif descriptor == 'akaze':
            kpts, desc = akaze_model.detectAndCompute(img, None)
            kpts = np.array([kp.pt for kp in kpts])
            desc = np.array(desc)

        out, N_matches = tracker.update(img, kpts, desc, descriptor)
        if descriptor in ['alike', 'aliked']:
            runtime.append(pred['time'])
            ave_fps = (1. / np.stack(runtime)).mean()
            status = f"Fps:{ave_fps:.1f}, Keypoints/Matches: {len(kpts)}/{N_matches}"
        else:
            status = f"Keypoints/Matches: {len(kpts)}/{N_matches}"
        progress_bar.set_description(status)

        if not args.no_display:
            cv2.setWindowTitle(args.model, args.model + ': ' + status)
            cv2.imshow(args.model, out)
            if cv2.waitKey(1) == ord('q'):
                break

    logging.info('Finished!')
    if not args.no_display:
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
        elif descriptor == 'sift':
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ALike Demo.')
    parser.add_argument('input', type=str, default='',
                        help='Image directory or movie file or "camera0" (for webcam0).')
    parser.add_argument('--model', choices=['alike-t', 'alike-s', 'alike-n', 'alike-l', "test"], default="alike-n",
                        help="The model configuration")
    parser.add_argument('--descriptor', choices=['alike', 'orb', 'sift', 'aliked', 'akaze'], default='alike')
    parser.add_argument('--n_limit', type=int, default=1000,
                        help='Maximum number of keypoints to be detected (default: 5000).')
    parser.add_argument('--no_display', action='store_true',
                        help='Do not display images to screen. Useful if running remotely (default: False).')
    parser.add_argument('--no_sub_pixel', action='store_true',
                        help='Do not detect sub-pixel keypoints (default: False).')
    args = parser.parse_args()

    main(args)