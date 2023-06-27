import sys
import os
import cv2
import numpy as np
import argparse
import image_resize


def eqr2dualfisheye(path="", output_size=(1024, 512)):

    eqr_image = cv2.imread(path)
    eqr_image = cv2.resize(eqr_image, output_size)
    eqr_img_h, eqr_img_w = eqr_image.shape[:2]

    f = eqr_img_h / np.pi  # focal length

    # RICHOの360度カメラ THETA Vのサイズ比に合わせてみた
    THETA_V_fisheye_img_h, THETA_V_fisheye_img_w = output_size
    THETA_V_eqr_img_h, THETA_V_eqr_img_w = output_size

    fisheye_img_h = -(- eqr_img_h * THETA_V_fisheye_img_h // THETA_V_eqr_img_h)
    fisheye_img_w = -(- eqr_img_w * THETA_V_fisheye_img_w // THETA_V_eqr_img_w)

    c_fisheye_lc = fisheye_img_w // 4        # 左側魚眼の中心のColumn座標
    c_fisheye_rc = fisheye_img_w // 4 * 3    # 右側魚眼の中心のColumn座標
    c_fisheye_border = fisheye_img_w // 2    # 左右魚眼の境界
    r_fisheye_c = fisheye_img_h // 2         # 左右魚眼の中心のRow座標

    c_fisheye_map, r_fisheye_map = np.meshgrid(range(fisheye_img_w), range(fisheye_img_h))

    # 魚眼CR座標 -> 魚眼xy座標
    y_fisheye_map = fisheye_img_h - r_fisheye_map - r_fisheye_c
    x_fisheye_map = np.zeros((fisheye_img_h, fisheye_img_w))
    x_fisheye_map[:, :c_fisheye_border] = c_fisheye_map[:, :c_fisheye_border] - c_fisheye_lc
    x_fisheye_map[:, c_fisheye_border:] = c_fisheye_map[:, c_fisheye_border:] - c_fisheye_rc

    # 魚眼xy座標 -> 魚眼極座標
    phi_fisheye_map = np.zeros((fisheye_img_h, fisheye_img_w))
    phi_fisheye_map[:, :c_fisheye_border] = np.arctan2(y_fisheye_map[:, :c_fisheye_border], x_fisheye_map[:,:c_fisheye_border])
    phi_fisheye_map[:, c_fisheye_border:] = np.arctan2(y_fisheye_map[:, c_fisheye_border:], - x_fisheye_map[:,c_fisheye_border:])
    phi_fisheye_map = phi_fisheye_map % (2 * np.pi)
    theta_fisheye_map = np.sqrt(x_fisheye_map**2 + y_fisheye_map**2) / f
    theta_fisheye_map[:, c_fisheye_border:] = np.pi - theta_fisheye_map[:, c_fisheye_border:]

    # 魚眼極座標 -> パノラマ緯度経度座標 -> パノラマCR座標
    c_eqr_map = f * (np.arctan2(-np.sin(theta_fisheye_map) * np.cos(phi_fisheye_map), -np.cos(theta_fisheye_map)) % (2 * np.pi))
    r_eqr_map = f * np.arccos(np.sin(theta_fisheye_map) * np.sin(phi_fisheye_map))

    # 魚眼CR座標 -> パノラマCR座標の完成
    fisheye2eqr_map_c = c_eqr_map.astype('float32')
    fisheye2eqr_map_r = r_eqr_map.astype('float32')

    dual_fisheye_image = cv2.remap(eqr_image, fisheye2eqr_map_c, fisheye2eqr_map_r, cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

    return dual_fisheye_image


def main():
    parser = argparse.ArgumentParser(description = 'Tangent Plane')
    parser.add_argument('--path', default = "./data/Room/0")
    args = parser.parse_args()

    #path = args.path
    #print(path)
    #resized_img = image_resize.image_resize(path, rows=5376, cols=2688)
    #print(resized_img.shape)
    #cv2.imwrite("data/test_farm/0/O2.png", resized_img)
    #print(args.path + "/O2.png")
    dual_fisheye_image = eqr2dualfisheye("data/test_farm/0/O2.png")


    cv2.imwrite("data/test_farm/0/O3.png", dual_fisheye_image)
    ###
    cv2.imshow('Dual Fisheye Image', dual_fisheye_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()