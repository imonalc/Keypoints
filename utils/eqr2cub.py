from tqdm import tqdm
import torch
import cv2
import numpy as np
import math

OUTPUT_DIR = "./output/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_theta_torch(x, y):
    theta = torch.where(y < 0, (-1) * torch.atan2(y, x), 2 * math.pi - torch.atan2(y, x))
    return theta

def create_equirectangler_to_bottom_and_top_map(input_w, input_h, output_sqr, z):
    x, y = torch.meshgrid(torch.linspace(-output_sqr/2.0, output_sqr/2.0-1, output_sqr), 
                          torch.linspace(-output_sqr/2.0, output_sqr/2.0-1, output_sqr),indexing='ij')
    x, y = x.to(device), y.to(device)
    z = torch.tensor(z).to(device)
    #print(x.shape, z)
    
    rho = torch.sqrt(x**2 + y**2 + z**2)
    norm_theta = get_theta_torch(x, y) / (2 * math.pi)
    norm_phi = (math.pi - torch.acos(z / rho)) / math.pi
    ix = norm_theta * input_w
    iy = norm_phi * input_h

    # Boundary handling
    ix = torch.where(ix >= input_w, ix - input_w, ix)
    iy = torch.where(iy >= input_h, iy - input_h, iy)
    
    return ix.cpu().numpy(), iy.cpu().numpy()

def create_equirectangler_to_front_and_back_map(input_w, input_h, output_sqr, x):
    z, y = torch.meshgrid(torch.linspace(-output_sqr/2.0, output_sqr/2.0-1, output_sqr), 
                          torch.linspace(-output_sqr/2.0, output_sqr/2.0-1, output_sqr),indexing='ij')
    z, y = z.to(device), y.to(device)
    x = torch.tensor(x).to(device)
    
    rho = torch.sqrt(x**2 + y**2 + z**2)
    norm_theta = get_theta_torch(x, y) / (2 * math.pi)
    norm_phi = (math.pi - torch.acos(z / rho)) / math.pi
    ix = norm_theta * input_w
    iy = norm_phi * input_h

    ix = torch.where(ix >= input_w, ix - input_w, ix)
    iy = torch.where(iy >= input_h, iy - input_h, iy)
    
    return ix.cpu().numpy(), iy.cpu().numpy()

def create_equirectangler_to_left_and_right_map(input_w, input_h, output_sqr, y):
    x, z = torch.meshgrid(torch.linspace(-output_sqr/2.0, output_sqr/2.0-1, output_sqr), 
                          torch.linspace(-output_sqr/2.0, output_sqr/2.0-1, output_sqr),indexing='ij')
    x, z = x.to(device), z.to(device)
    y = torch.tensor(y).to(device)
    
    rho = torch.sqrt(x**2 + y**2 + z**2)
    norm_theta = get_theta_torch(x, y) / (2 * math.pi)
    norm_phi = (math.pi - torch.acos(z / rho)) / math.pi
    ix = norm_theta * input_w
    iy = norm_phi * input_h

    # Boundary handling
    ix = torch.where(ix >= input_w, ix - input_w, ix)
    iy = torch.where(iy >= input_h, iy - input_h, iy)
    
    return ix.cpu().numpy(), iy.cpu().numpy()


def create_cube_map(back_img, bottom_img, front_img, left_img, right_img, top_img, output_sqr):
    cube_map_img = np.zeros((3 * output_sqr, 4 * output_sqr, 3))
    cube_map_img[output_sqr:2*output_sqr, 3*output_sqr:4*output_sqr] = back_img
    cube_map_img[2*output_sqr:3*output_sqr, output_sqr:2*output_sqr] = bottom_img
    cube_map_img[output_sqr:2*output_sqr, output_sqr:2*output_sqr] = front_img
    cube_map_img[output_sqr:2*output_sqr, 0:output_sqr] = left_img
    cube_map_img[output_sqr:2*output_sqr, 2*output_sqr:3*output_sqr] = right_img
    cube_map_img[0:output_sqr, output_sqr:2*output_sqr] = top_img
    return cube_map_img


def create_cube_imgs(img):
    input_h, input_w, _ = img.shape
    output_sqr = int(input_w / 4)
    normalized_f = 1

    z = (output_sqr / (2.0 * normalized_f))
    bottom_map_x, bottom_map_y = create_equirectangler_to_bottom_and_top_map(input_w, input_h, output_sqr, z)
    bottom_img = cv2.remap(img, bottom_map_x.astype("float32"), bottom_map_y.astype("float32"), cv2.INTER_CUBIC)
    #bottom_img = cv2.rotate(bottom_img, cv2.ROTATE_90_CLOCKWISE)

    z = (-1) * (output_sqr / (2.0 * normalized_f))
    top_map_x, top_map_y = create_equirectangler_to_bottom_and_top_map(input_w, input_h, output_sqr, z)
    top_img = cv2.remap(img, top_map_x.astype("float32"), top_map_y.astype("float32"), cv2.INTER_CUBIC)
    top_img = cv2.flip(top_img, 0)
    #top_img = cv2.rotate(top_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    x = (-1) * (output_sqr / (2.0 * normalized_f))
    front_map_x, front_map_y = create_equirectangler_to_front_and_back_map(input_w, input_h, output_sqr, x)
    front_img = cv2.remap(img, front_map_x.astype("float32"), front_map_y.astype("float32"), cv2.INTER_CUBIC)

    x = output_sqr / (2.0 * normalized_f)
    back_map_x, back_map_y = create_equirectangler_to_front_and_back_map(input_w, input_h, output_sqr, x)
    back_img = cv2.remap(img, back_map_x.astype("float32"), back_map_y.astype("float32"), cv2.INTER_CUBIC)
    back_img = cv2.flip(back_img, 1)

    y = (-1) * (output_sqr / (2.0 * normalized_f))
    left_map_x, left_map_y = create_equirectangler_to_left_and_right_map(input_w, input_h, output_sqr, y)
    left_img = cv2.remap(img, left_map_x.astype("float32"), left_map_y.astype("float32"), cv2.INTER_CUBIC)
    left_img = cv2.rotate(left_img, cv2.ROTATE_90_CLOCKWISE)

    y = output_sqr / (2.0 * normalized_f)
    right_map_x, right_map_y = create_equirectangler_to_left_and_right_map(input_w, input_h, output_sqr, y)
    right_img = cv2.remap(img, right_map_x.astype("float32"), right_map_y.astype("float32"), cv2.INTER_CUBIC)
    right_img = cv2.flip(right_img, 1)
    right_img = cv2.rotate(right_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return [back_img, bottom_img, front_img, left_img, right_img, top_img], output_sqr


def main(image_path):
    img = cv2.imread(image_path)
    print(img.shape, type(img))
    [back_img, bottom_img, front_img, left_img, right_img, top_img], output_sqr = create_cube_imgs(img)
    cube_map_img = create_cube_map(back_img, bottom_img, front_img, left_img, right_img, top_img, output_sqr)
    
    cv2.imwrite(f"{OUTPUT_DIR}bottom.png", bottom_img)
    cv2.imwrite(f"{OUTPUT_DIR}top.png", top_img)
    cv2.imwrite(f"{OUTPUT_DIR}front.png", front_img)
    cv2.imwrite(f"{OUTPUT_DIR}back.png", back_img)
    cv2.imwrite(f"{OUTPUT_DIR}left.png", left_img)
    cv2.imwrite(f"{OUTPUT_DIR}right.png", right_img)
    cv2.imwrite(f"{OUTPUT_DIR}cube_map.png", cube_map_img)
    return


if __name__ == "__main__":
    image_path = "O.jpg"
    main(image_path)
