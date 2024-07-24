/*
Mesh ray intersection
Written by Xiaohu Lu
All Rights Reserved.
*/

#include <iostream>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <assert.h>


//#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
//#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
//#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


void mesh_ray_intersection_launcher(int vertexes_num, int faces_num, int rays_num, const float *vertexes, 
const int *faces, const float *ray_origins, const float *ray_directions, float *result);

int mesh_ray_intersection_gpu(at::Tensor vertexes_tensor, at::Tensor faces_tensor, 
at::Tensor ray_origins_tensor, at::Tensor ray_directions_tensor, at::Tensor result_tensor){
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (B, npoints, 3) [x, y, z]
    // params boxes_idx_of_points: (B, npoints), default -1

//    CHECK_INPUT(boxes_tensor);
//    CHECK_INPUT(pts_tensor);
//    CHECK_INPUT(box_idx_of_points_tensor);
    
    int vertexes_num = vertexes_tensor.size(0);
    int faces_num = faces_tensor.size(0);
    int rays_num = ray_origins_tensor.size(0);

    const float *vertexes = vertexes_tensor.data<float>();
    const int *faces = faces_tensor.data<int>();
    const float *ray_origins = ray_origins_tensor.data<float>();
    const float *ray_directions = ray_directions_tensor.data<float>();
    float *result = result_tensor.data<float>();
    mesh_ray_intersection_launcher(vertexes_num, faces_num, rays_num, vertexes, faces, ray_origins, ray_directions, result);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mesh_ray_intersection_gpu", &mesh_ray_intersection_gpu, "mesh ray intersection (CUDA)");
}
