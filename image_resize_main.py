import sys
import os
sys.path.append(os.getcwd()+'/fivepoint')
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
    print(path_o)

    img_o = cv2.imread(path_o)
    img_r = cv2.imread(path_r)

    resized_img_o = cv2.resize(img_o, (1280, 640))
    resized_img_r = cv2.resize(img_r, (1280, 640))

    cv2.imwrite(args.path + "/O2.png", resized_img_o)
    cv2.imwrite(args.path + "/R2.png", resized_img_r)

if __name__ == '__main__':
    main()