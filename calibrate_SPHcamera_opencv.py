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



def calib_sph(data):

    raw = (3840, 1920)
    resized = (1024, 512)

    # チェスボードの格子点の数
    pattern_size = (7, 8)
    square_size = 6.5

    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    pattern_points = np.zeros((1, pattern_size[0]*pattern_size[1], 3), np.float32)
    pattern_points[0,:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # 格子点を検出するための変数
    obj_points = []  # 実世界座標
    img_points1 = []  # 画像1の座標
    img_points2 = []  # 画像2の座標

    # 画像ファイルのパス
    image1_path = os.path.join(os.getcwd(), "data/Calibration", data, 'O.jpg')
    image2_path = os.path.join(os.getcwd(), "data/Calibration", data, 'R.jpg')

    print(image1_path)

    # 画像1の読み込み
    image1 = cv2.imread(image1_path)
    image1 = eqr2dualfisheye(image1, output_size=raw, side="right")
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # 画像2の読み込み
    image2 = cv2.imread(image2_path)
    image2 = eqr2dualfisheye(image2, output_size=raw, side="right")
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("Modified Image", gray2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # チェスボードの格子点を検出
    found1, corners1 = cv2.findChessboardCorners(gray1, pattern_size)
    found2, corners2 = cv2.findChessboardCorners(gray2, pattern_size)

    # 格子点が検出された場合
    if found1 and found2:
        # サブピクセル単位での格子点の位置を検出
        cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
        cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))

        # 格子点を追加
        obj_points.append(pattern_points)
        img_points1.append(corners1)
        img_points2.append(corners2)

        # 魚眼カメラキャリブレーションの実行
        K1 = np.zeros((3, 3))
        K2 = np.zeros((3, 3))
        D1 = np.zeros((4, 1))
        D2 = np.zeros((4, 1))


        _, K1, D1, _, _ = cv2.fisheye.calibrate(
            obj_points, img_points1, gray1.shape[::-1], K1, D1, None, None, flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        )
        _, K2, D2, _, _ = cv2.fisheye.calibrate(
            obj_points, img_points2, gray2.shape[::-1], K2, D2, None, None, flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        )

        # カメラ行列の逆行列を計算
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)

        # 2つの視点間の外部パラメータを推定
        flags = cv2.CALIB_FIX_INTRINSIC + cv2.CALIB_USE_INTRINSIC_GUESS
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        ret= cv2.stereoCalibrate(
            obj_points, 
            img_points1, 
            img_points2, 
            K1,
            D1, 
            K2, 
            D2, 
            gray1.shape[::-1],
            flags=flags,
            criteria=criteria
        )
        err, KKLeft, distCoeffsLeft, KKRight, distCoeffsRight, R, T, E, F = ret

        # 並進ベクトルをカメラ行列の逆行列で変換
        #T1 = np.dot(K1_inv, T[0])
        #T2 = np.dot(K2_inv, T[1])

        # 外部パラメータの表示
        #print("カメラ1行列:\n", K1)
        #print("カメラ1の歪み係数:\n", D1)
        #print("カメラ2行列:\n", K2)
        #print("カメラ2の歪み係数:\n", D2)


        ## 画像1を保存
        #cv2.imwrite('image1_corners.jpg', image1)
#
        ## 画像2を保存
        #cv2.imwrite('image2_corners.jpg', image2)
#
        ## 画像1と画像2の格子点を描画したものを表示
        #cv2.imshow('Image 1 with Corners', image1)
        #cv2.imshow('Image 2 with Corners', image2)
        #cv2.waitKey(0)
        return R, T
    else:
        #print("チェスボードの格子点が検出されませんでした。")
        R = np.array([[ 0, 0, 0],
            [0, 0, 0],
            [0,  0, 0]])
        T = np.array([0,0,0])
        return R, T


def main():

    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--data'      , default="pose0/0")
    args = parser.parse_args()
    R, T = calib_sph(args.data)
    

if __name__ == '__main__':
    main()