import sys
import os
import cv2
import numpy as np
import argparse


def eqr2dualfisheye(img, output_size=(1024, 512), side="right"):
    img_resized = cv2.resize(img, output_size)
    if side == "right":
        img_eqr = cut_leftside(img_resized)
    elif side == "left":
        img_eqr = cut_rightside(img_resized)
    else:
        print("error")
        exit()
    h_img_eqr, w_img_eqr = img_eqr.shape[:2]
    h_img_Dfish, w_img_Dfish = img_eqr.shape[:2]
    f = h_img_eqr / np.pi

    Oy_img_fish = h_img_eqr // 2
    lOx_img_fish = w_img_eqr // 4
    rOx_img_fish = w_img_eqr * 3 // 4
    boundaryx_img_Dfish = w_img_eqr * 2 // 4

    w_map_Dfish, h_map_Dfish = np.meshgrid(range(w_img_Dfish), range(h_img_Dfish))

    y_map_fish = h_img_Dfish - h_map_Dfish - Oy_img_fish
    x_map_fish = np.zeros((h_img_Dfish, w_img_Dfish))
    x_map_fish[:, :boundaryx_img_Dfish] = w_map_Dfish[:, :boundaryx_img_Dfish] - lOx_img_fish
    x_map_fish[:, boundaryx_img_Dfish:] = w_map_Dfish[:, boundaryx_img_Dfish:] - rOx_img_fish

    theta_map_fish = np.sqrt(x_map_fish**2 + y_map_fish**2) / f
    theta_map_fish[:, boundaryx_img_Dfish:] = np.pi - theta_map_fish[:, boundaryx_img_Dfish:]
    phi_map_fish = np.zeros((h_img_Dfish, w_img_Dfish))
    phi_map_fish[:, :boundaryx_img_Dfish] = np.arctan2(y_map_fish[:, :boundaryx_img_Dfish], x_map_fish[:,:boundaryx_img_Dfish])
    phi_map_fish[:, boundaryx_img_Dfish:] = np.arctan2(y_map_fish[:, boundaryx_img_Dfish:], - x_map_fish[:,boundaryx_img_Dfish:])
    phi_map_fish = phi_map_fish % (2 * np.pi)

    y_map_eqr = f * (np.arctan2(-np.sin(theta_map_fish) * np.cos(phi_map_fish), -np.cos(theta_map_fish)) % (2 * np.pi))
    x_map_eqr = f * np.arccos(np.sin(theta_map_fish) * np.sin(phi_map_fish))

    y_map_fisheye2eqr = y_map_eqr.astype('float32')
    x_map_fisheye2eqr = x_map_eqr.astype('float32')

    dual_fisheye_image = cv2.remap(img_eqr, y_map_fisheye2eqr, x_map_fisheye2eqr, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    if side == "right":
        ret = dual_fisheye_image[:, :w_img_Dfish//2]
    else:
        ret = dual_fisheye_image[:, w_img_Dfish//2:w_img_Dfish]

    return ret


def cut_rightside(img):
    print(img.shape)
    h_img, w_img = img.shape[:2]
    st_col = w_img // 4
    ed_col = w_img * 3 // 4
    left_img = img
    left_img[:, st_col:ed_col] = (0, 0, 0)

    #cv2.imshow("Modified Image", left_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return left_img

def cut_leftside(img):
    print(img.shape)
    h_img, w_img = img.shape[:2]
    st_col = w_img // 4
    ed_col = w_img * 3 // 4
    right_img = img
    right_img[:, 0:st_col] = (0, 0, 0)
    right_img[:, ed_col: w_img] = (0, 0, 0)

    #cv2.imshow("Modified Image", left_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return right_img



#def main():
#    parser = argparse.ArgumentParser(description = 'Tangent Plane')
#    parser.add_argument('--path', default = "./data/test_farm/0/O.png")
#    args = parser.parse_args()
#
#    #path = args.path
#    #print(path)
#    #resized_img = image_resize.image_resize(path, rows=5376, cols=2688)
#    #print(resized_img.shape)
#    #cv2.imwrite("data/test_farm/0/O2.png", resized_img)
#    #print(args.path + "/O2.png")
#    #dual_fisheye_image = eqr2dualfisheye("data/test_farm/0/O2.png")
#    img_eqr = cv2.imread(args.path)
#    dual_fisheye_image = eqr2dualfisheye(img_eqr)
#    h_dual_fisheye_image, w_dual_fisheye_image = dual_fisheye_image.shape[:2]
#    right_fisheye_image = dual_fisheye_image[:, w_dual_fisheye_image //2:w_dual_fisheye_image]
#    ###
#    cv2.imshow('Dual Fisheye Image', right_fisheye_image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
#
#if __name__ == '__main__':
#    main()