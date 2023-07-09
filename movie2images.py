import sys
import os
sys.path.append(os.getcwd()+'/fivepoint')
import build.fivep as f

import time
import torch
import torch.nn.functional as F
from spherical_distortion.functional import create_tangent_images, unresample
from spherical_distortion.util import *
import matplotlib.pyplot as plt
from skimage import io
import os
import cv2

import sys
import pandas as pd
import numpy as np
import _spherical_distortion_ext._mesh as _mesh
import argparse

from random import sample
import imageio
from scipy.spatial.transform import Rotation as Rot

from utils.coord    import coord_3d
from utils.ransac   import *
from utils.keypoint import *
from utils.metrics  import *
from utils.eqr_to_dualfisheyes import *
from utils.camera_recovering import *

from os import listdir
from os.path import isfile, join, isdir
from tqdm import tqdm
from scipy.spatial.transform import Rotation

def extract_frames(video_path, output_path, frame_interval):
    # 動画ファイルを開く
    video_capture = cv2.VideoCapture(video_path)
    
    # フレームカウンタを初期化
    frame_count = 0
    
    while True:
        # フレームを読み込む
        ret, frame = video_capture.read()
        
        # フレームが正常に読み込まれなかった場合、ループを終了
        if not ret:
            break
        
        # フレームカウンタが指定したフレーム間隔の倍数のときに画像を保存
        if frame_count % frame_interval == 0:
            output_filename = f"{output_path}/frame_{frame_count}.jpg"
            cv2.imwrite(output_filename, frame)
        
        # フレームカウンタをインクリメント
        frame_count += 1
    
    # メモリを解放し、ファイルを閉じる
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    # 使用例
    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--data'      , default="Calibration/pose1")
    args = parser.parse_args()


    video_path = os.path.join(os.getcwd(), "data", args.data, 'O.MP4')
    output_path = os.path.join(os.getcwd(), "data", args.data, 'images')    # 出力画像の保存先ディレクトリパス
    frame_interval = 100               # 画像を切り出すフレーム間隔

    extract_frames(video_path, output_path, frame_interval)


if __name__ == '__main__':
    main()