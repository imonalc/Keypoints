import sys
import os
sys.path.append(os.getcwd()+'/fivepoint')
import cv2

import sys

def image_resize(path="/data/Room/0", rows=640, cols=1280):
    path = os.getcwd()+path+"/O.png"
    img = cv2.imread(path)
    resized_img_o = cv2.resize(img, (rows, cols))
    return resized_img_o