"""
Written by Xiaohu Lu
All Rights Reserved 2019-2020.
"""
import torch

from ...utils import common_utils
from . import mesh_ray_intersection_cuda

def mesh_ray_intersection_3d_cuda(vertexes, faces, ray_origins, ray_directions):
    vertexes, is_numpy = common_utils.check_numpy_to_torch(vertexes)
    faces, is_numpy = common_utils.check_numpy_to_torch(faces)
    ray_origins, is_numpy = common_utils.check_numpy_to_torch(ray_origins)
    ray_directions, is_numpy = common_utils.check_numpy_to_torch(ray_directions)

    assert faces.shape[1] == 3
    intesection_dis = torch.cuda.FloatTensor(torch.Size((ray_directions.shape[0], 1))).zero_()
    mesh_ray_intersection_cuda.mesh_ray_intersection_gpu(vertexes.contiguous(), 
    faces.contiguous(), ray_origins.contiguous(), ray_directions.contiguous(), intesection_dis)
    return intesection_dis.numpy() if is_numpy else intesection_dis

