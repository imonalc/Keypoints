import torch

def cube_to_equirectangular_coord(face: str, cor_xy: (float, float), width: int) -> (float, float):
    vec = cube_coord_to_3d_vector(face, cor_xy, width)
    return vector_to_equirectangular_coord(vec)

def cube_coord_to_3d_vector(face: str, cor_xy: (float, float), width: int) -> torch.Tensor:
    x, y = cor_xy
    half_width = width / 2.0
    
    # Convert 2D coordinates to range [-1, 1]
    x = (x / half_width) - 1
    y = (y / half_width) - 1
    
    # Depending on the face, compute the 3D direction
    if face == "front":
        return torch.tensor([1.0, -x, -y])
    elif face == "back":
        return torch.tensor([-1.0, x, -y])
    elif face == "right":
        return torch.tensor([x, 1.0, -y])
    elif face == "left":
        return torch.tensor([-x, -1.0, -y])
    elif face == "top":
        return torch.tensor([x, y, 1.0])
    elif face == "bottom":
        return torch.tensor([x, -y, -1.0])
    else:
        raise ValueError(f"Invalid face name: {face}")
    
def vector_to_equirectangular_coord(vec: torch.Tensor) -> (float, float):
    # Convert 3D cartesian coordinates to spherical coordinates
    r = torch.sqrt(torch.sum(vec**2))
    theta = torch.acos(vec[2] / r)  # polar angle (0 <= theta <= pi)
    phi = torch.atan2(vec[1], vec[0])  # azimuthal angle (-pi <= phi <= pi)
    
    # Convert spherical coordinates to 2D equirectangular coordinates
    # Assuming equirectangular image has width 2 * height
    x = (phi + torch.pi) / (2 * torch.pi)
    y = theta / torch.pi
    
    return x, y


if __name__ == "__main__":
    # Test the function
    test_face = "front"
    test_cor_xy = (256, 256)
    test_width = 512

    print(cube_to_equirectangular_coord(test_face, test_cor_xy, test_width))