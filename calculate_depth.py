import sys
import os
sys.path.append(os.getcwd()+'/fivepoint')
import os
import cv2

import sys
import pandas as pd
import numpy as np
import argparse

from tqdm import tqdm

def main():

    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--path', default = "./data/Room/0")
    args = parser.parse_args()

    
    print('X')

    path_o = args.path + '/O.png'
    path_r = args.path + '/R.png'
    

    img_o = cv2.imread(path_o)
    img_r = cv2.imread(path_r)
    Rx = np.load(args.path+"/R.npy")

    print(Rx)



    # 画像の深度を計算する
    depth = img_o.shape[2]  # 画像のチャンネル数

    print('画像の深度:', depth)

if __name__ == '__main__':
    main()