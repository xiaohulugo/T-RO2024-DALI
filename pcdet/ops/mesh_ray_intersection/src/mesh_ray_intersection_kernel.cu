/*
Möller–Trumbore intersection algorithm CUDA
Written by Xiaohu Lu
All Rights Reserved 2022-2023.
*/

#include <iostream>
#include <math.h>
#include <stdio.h>

#define INFIN 100000
#define THREADS_PER_BLOCK 128
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
// #define DEBUG

__device__ inline void vec3_sub(float *v1, float *v0, float *result){
    result[0] = v1[0]-v0[0];
    result[1] = v1[1]-v0[1];
    result[2] = v1[2]-v0[2];
}

__device__ inline void vec3_cross(float *a, float *b, float *result){
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ inline float vec3_dot(float *a, float *b){
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

__device__ inline float mesh_ray_intersection(const float *r_ori, const float *r_dir, const float *v0, const float *v1, const float *v2){        
    float v01[3], v02[3], pvec[3], tvec[3], qvec[3];    
    v01[0] = v1[0] - v0[0];
    v01[1] = v1[1] - v0[1];
    v01[2] = v1[2] - v0[2];

    v02[0] = v2[0] - v0[0];
    v02[1] = v2[1] - v0[1];
    v02[2] = v2[2] - v0[2];    

    pvec[0] = r_dir[1] * v02[2] - r_dir[2] * v02[1];
    pvec[1] = r_dir[2] * v02[0] - r_dir[0] * v02[2];
    pvec[2] = r_dir[0] * v02[1] - r_dir[1] * v02[0];    

    float det = v01[0] * pvec[0] + v01[1] * pvec[1] + v01[2] * pvec[2];  
    if (det < 0.000001)
        return -INFIN;    

    float inv_det = 1.0 / det;
    tvec[0] = r_ori[0] - v0[0];
    tvec[1] = r_ori[1] - v0[1];
    tvec[2] = r_ori[2] - v0[2];

    float u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2])*inv_det;
    if (u < 0 || u > 1)
        return -INFIN;

    qvec[0] = tvec[1] * v01[2] - tvec[2] * v01[1];
    qvec[1] = tvec[2] * v01[0] - tvec[0] * v01[2];
    qvec[2] = tvec[0] * v01[1] - tvec[1] * v01[0];
    float v = (r_dir[0] * qvec[0] + r_dir[1] * qvec[1] + r_dir[2] * qvec[2])*inv_det;
    if (v < 0 || u + v > 1)
        return -INFIN;
    
    float dis = (v02[0] * qvec[0] + v02[1] * qvec[1] + v02[2] * qvec[2])*inv_det;
    if(dis > 0.000001){
        return dis;
    }else{
        return -INFIN;
    }
}


__global__ void mesh_ray_intersection_kernel(int vertexes_num, int faces_num, int rays_num, const float *vertexes, 
const int *faces, const float *ray_origins, const float *ray_directions, float *result){
    // params vertexes: (N, 3) [x, y, z]
    // params faces: (N, 3) [id_vertex1, id_vertex2, id_vertex3]
    // params ray_origins: (M, 3) [x, y, z]
    // params ray_directions: (M, 3) [rx, ry, rz]

    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= rays_num) return;

    const float* r_ori = ray_origins + 3*ray_idx;
    const float* r_dir = ray_directions + 3*ray_idx;
    float dis_min = INFIN;
    
    for (int k = 0; k < faces_num; k++){
        const float* v0 = vertexes + 3 * faces[3*k+0];
        const float* v1 = vertexes + 3 * faces[3*k+1];
        const float* v2 = vertexes + 3 * faces[3*k+2];              

        float dis = mesh_ray_intersection(r_ori, r_dir, v0, v1, v2);        
        if (dis != -INFIN && dis < dis_min){
            dis_min = dis;
        }
    }
    result[ray_idx] = dis_min;
}


void mesh_ray_intersection_launcher(int vertexes_num, int faces_num, int rays_num, const float *vertexes, 
const int *faces, const float *ray_origins, const float *ray_directions, float *result){
    // params vertexes_num: (N, 3) [x, y, z]
    // params faces: (N, 3) [id_vertex1, id_vertex2, id_vertex3]
    // params ray_origins: (M, 3) [x, y, z]
    // params ray_directions: (M, 3) [rx, ry, rz]
    cudaError_t err;

    dim3 blocks(DIVUP(rays_num, THREADS_PER_BLOCK), 1);
    dim3 threads(THREADS_PER_BLOCK);
    mesh_ray_intersection_kernel<<<blocks, threads>>>(vertexes_num, faces_num, rays_num, vertexes, faces, ray_origins,ray_directions,result);
    
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}
