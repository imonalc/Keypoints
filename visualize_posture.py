import open3d as o3d
import numpy as np

axis_a = o3d.geometry.TriangleMesh.create_coordinate_frame()
axis_b = o3d.geometry.TriangleMesh.create_coordinate_frame()
T_a = np.eye(4) # Camera pose A
T_b = np.array(
    [
        [-0.03376933, -0.99942672, -0.00241938, 0.79321197],
        [ 0.99942374, -0.03377736,  0.0033578 , -0.60761616],
        [-0.00343759, -0.00230459,  0.99999144, 0.0402166],
        [0, 0, 0, 1]
    ]
) # Camera pose B
axis_a.transform(T_a)
axis_b.transform(T_b)
o3d.visualization.draw_geometries([axis_a, axis_b])