import torch

def cube_coord_to_3d_vector(face: str, cor_xy: (float, float), width: int) -> torch.Tensor:
    x, y = cor_xy
    x -= width
    y -= width

    
    # Depending on the face, compute the 3D direction
    if face == "front":
        return torch.tensor([width, y, -x])#torch.tensor([x, y, width/2])
    elif face == "back":
        return torch.tensor([-width, -y, -x])
    elif face == "right":
        return torch.tensor([-y, width, -x])#torch.tensor([x, width/2, y])
    elif face == "left":
        return torch.tensor([y, -width, -x])#torch.tensor([width/2, y, x])
    elif face == "top":
        return torch.tensor([x, y, width])#torch.tensor([x, y, width/2])#torch.tensor([x, y, -width/2]) # ok
    elif face == "bottom":
        return torch.tensor([-x, y, -width])#torch.tensor([x, y, -width/2])
    else:
        raise ValueError(f"Invalid face name: {face}")
    
def vector_to_equirectangular_coord(vec: torch.Tensor, width: int) -> (float, float):
    # Convert 3D cartesian coordinates to spherical coordinates
    r = torch.sqrt(torch.sum(vec**2))
    theta = torch.acos(vec[2] / r)  # polar angle (0 <= theta <= pi)
    phi = torch.atan2(vec[1], vec[0])  # azimuthal angle (-pi <= phi <= pi)
    
    # Convert spherical coordinates to 2D equirectangular coordinates
    # Taking into account the new desired width and height
    x = width * 8 * (phi + torch.pi) / (2 * torch.pi)
    y = width * 4 * theta / torch.pi
    
    return x.item(), y.item()

def cube_to_equirectangular_coord(face: str, cor_xy: (float, float), width: int) -> (float, float):
    vec = cube_coord_to_3d_vector(face, cor_xy, width)
    #print(vector_to_equirectangular_coord(vec, width))
    return vector_to_equirectangular_coord(vec, width)


if __name__ == "__main__":
    # Test the function
    test_face = "front"
    test_cor_xy = (256, 256)
    test_width = 512

    print(cube_to_equirectangular_coord(test_face, test_cor_xy, test_width))