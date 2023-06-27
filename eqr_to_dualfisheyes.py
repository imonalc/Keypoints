import sys
import os
import cv2
import numpy as np
import argparse
import image_resize


def eqr2dualfisheye(path="./data/test_farm/0/O.png", output_size=(1024, 512)):
    img_eqr = cv2.imread(path)
    img_eqr = cv2.resize(img_eqr, output_size)
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

    return dual_fisheye_image


def main():
    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--path', default = "./data/test_farm/0/O.png")
    args = parser.parse_args()

    #path = args.path
    #print(path)
    #resized_img = image_resize.image_resize(path, rows=5376, cols=2688)
    #print(resized_img.shape)
    #cv2.imwrite("data/test_farm/0/O2.png", resized_img)
    #print(args.path + "/O2.png")
    #dual_fisheye_image = eqr2dualfisheye("data/test_farm/0/O2.png")
    dual_fisheye_image = eqr2dualfisheye(args.path)
    ###
    cv2.imshow('Dual Fisheye Image', dual_fisheye_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()