import glob, os
from matplotlib.pyplot import axis, box
import numpy as np
import trimesh
from vedo import Points, Lines, show
import copy
import torch
import random
import math
import time
from numpy import linalg as LA
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from pcdet.ops.mesh_ray_intersection import mesh_ray_intersection_utils
from pcdet.utils import box_utils


def as_mesh(scene):
    dump = scene.dump() 
    mesh_list = []   
    for mesh in dump:
        if mesh.faces.shape[1] !=3:
            continue
        mesh_list.append(mesh)
    
    mesh = trimesh.util.concatenate(tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)for g in mesh_list))
    return mesh

def generate_rays(origin, beam_list, numpts_per_beam):
    longitude_list = np.linspace(0, 2*np.pi, numpts_per_beam)
    altitude_list = beam_list

    ray_directions = []
    for alt in altitude_list:
        z_list = np.sin(alt)*np.ones(len(longitude_list))
        x_list = np.cos(alt)*np.cos(longitude_list)
        y_list = np.cos(alt)*np.sin(longitude_list)
        xyz_list = np.vstack([x_list, y_list, z_list])
        ray_directions.append(np.transpose(xyz_list))
    ray_directions = np.concatenate(ray_directions)
    ray_origins = np.repeat(np.array(origin).reshape(1,-1), ray_directions.shape[0], axis=0)
    
    return ray_origins, ray_directions



def mesh_ray_tracing_point1(pts_ref, ray_origin, pts, bbox=None):           
    # align pts_ref with bbox
    angle = bbox[6]
    Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    pts_ref[:,0:3] = np.transpose(np.matmul(Rz, np.transpose(pts_ref[:,0:3], (1,0))), (1,0))  
    pts_ref[:,0:3] += bbox[0:3]
    
    # get bbox rays
    ray_pts = pts - ray_origin
    norm_pts = np.linalg.norm(ray_pts, axis=1)
    ray_pts /= norm_pts[:,None]

    # get pts_ref  rays
    ray_refpts = pts_ref - ray_origin
    norm_refpts = np.linalg.norm(ray_refpts, axis=1)
    ray_refpts /= norm_refpts[:,None]

    # calculate ray-angle deviation
    dis = cdist(ray_pts, ray_refpts, 'cosine')
    order = np.argsort(dis, axis=1)
    
    # find best ref_pts
    k = 10
    pts_new = []
    index = np.arange(pts_ref.shape[0])
    for i in range(pts.shape[0]):
        index_select = order[i,0:k]
        depth_select = norm_refpts[index_select]
        idx = np.argmin(depth_select)
        index_best = index_select[idx]
        pts_new.append(pts_ref[index_best])
    pts_new = np.vstack(pts_new)

    return pts_new, None



def mesh_ray_tracing_point2(pts_ref, ray_origin, shift, ray_directions, bbox=None):         
    # align pts_ref with bbox
    angle = bbox[6]
    Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    pts_ref[:,0:3] = np.transpose(np.matmul(Rz, np.transpose(pts_ref[:,0:3], (1,0))), (1,0))  
    pts_ref[:,0:3] += bbox[0:3]

    # move to remote
    pts_ref[:,0:3] += shift

    #
    pts_ref_mean = np.mean(pts_ref, axis=0)
    dis_mean = (pts_ref_mean[0]**2+pts_ref_mean[1]**2)**(0.5)
    th_cosine = 1.0-np.cos(0.1/dis_mean)

    # get pts_ref  rays
    ray_refpts = pts_ref - ray_origin
    norm_refpts = np.linalg.norm(ray_refpts, axis=1)
    ray_refpts /= norm_refpts[:,None]

    # calculate ray-angle deviation
    tt = np.random.permutation(ray_refpts.shape[0])
    ray_refpts_mini = ray_refpts[tt[0:int(tt.shape[0]*0.2)]]
    dis_mini = cdist(ray_directions, ray_refpts_mini, 'cosine')
    count_mini = np.sum(dis_mini < 3*th_cosine, axis=1)   
    ray_index = np.arange(ray_directions.shape[0]) 
    ray_index_mini = ray_index[count_mini>0]
    ray_directions_mini = ray_directions[count_mini>0]

    dis = cdist(ray_directions_mini, ray_refpts, 'cosine')
    count = np.sum(dis < th_cosine, axis=1)    
    ray_index_final = ray_index_mini[count>0] 
    dis_final = dis[count>0] 
    
    # find best ref_pts
    pts_list = []
    rays_list = []
    index = np.arange(pts_ref.shape[0])    
    for i in range(ray_index_final.shape[0]):  
        id_ray = ray_index_final[i]
        dis_i = dis_final[i,:]
        mask_i = dis_i < th_cosine
        if np.sum(mask_i)==0:
            continue
        idx_i = index[mask_i]
        depth_i = norm_refpts[mask_i]

        idx = np.argmin(depth_i)
        index_best = idx_i[idx]
        pts_list.append(pts_ref[index_best])
        rays_list.append(id_ray)
    if len(pts_list):
        pts_new = np.vstack(pts_list)
        rays_new = np.vstack(rays_list)
    else:
        return None, None

    if 0:
        pc1 = Points(pts_new, r=7, c='red')   
        pc3 = Points(pts_ref, r=5, c='blue')   
        show([(pc1, pc3)], N=2, axes=1)          
        a = 0 

    pts_new[:,0:3] -= shift
    return pts_new, rays_new



def mesh_ray_tracing_mesh4(mesh, cad_lwh, ray_origin, ray_directions, bbox=None, step_size=50, points=None):           
    #mesh = as_mesh(mesh)
    #faces = mesh.faces
    #faces_area = mesh.area_faces
    #mesh.faces = faces[faces_area>0.001]
    
    ray_directions[0,:] = ray_origin-bbox[0:3]
    # transform from car system to cad system
    # rotate z
    angle = -bbox[6]
    Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    ray_directions = np.transpose(np.matmul(Rz, np.transpose(ray_directions, (1,0))), (1,0))    

    # rotate x
    Rx = np.array([[1,0,0],[0,np.cos(-np.pi/2.0),-np.sin(-np.pi/2.0)],[0,np.sin(-np.pi/2.0),np.cos(-np.pi/2.0)]])
    ray_directions = np.transpose(np.matmul(Rx, np.transpose(ray_directions, (1,0))), (1,0))     
    
    # translate
    ray_directions[0,1] += cad_lwh[2]/2.0

    ray_origin_transformed = ray_directions[0,:]
    num_pts = ray_directions.shape[0]-1
    ray_origins = torch.tensor(ray_origin_transformed).cuda().float().view(1,-1).repeat(num_pts, 1)      
    ray_directions = torch.tensor(ray_directions[1:,:]).cuda().float()

    # intersection
    vertexes = torch.tensor(mesh.vertices).cuda().float()
    faces = torch.tensor(mesh.faces).cuda().int()
    ray_intersection_dis = mesh_ray_intersection_utils.mesh_ray_intersection_3d_cuda(vertexes, faces, ray_origins, ray_directions)

    # get intersection point
    ray_origins = ray_origins.cpu().numpy()
    ray_directions = ray_directions.cpu().numpy()
    ray_intersection_dis = ray_intersection_dis.cpu().numpy()
    locations_all = ray_origins+ray_directions*ray_intersection_dis
    index_ray_all = np.arange(0, ray_directions.shape[0], 1, dtype=int)
    mask_ray = ray_intersection_dis[:,0]<1000
    if not np.sum(mask_ray):
        return None, None
    locations_all = locations_all[mask_ray]
    index_ray_all = index_ray_all[mask_ray]

    # locations_all = []
    # index_ray_all = []
    # for i in range(ray_directions.shape[0]):
    #     if ray_intersection_dis[i] < 1000:
    #         pt = ray_origins[i,:]+ray_directions[i,:]*ray_intersection_dis[i]
    #         locations_all.append(pt.reshape(1,3))
    #         index_ray_all.append(i)
    # if not len(locations_all):
    #     return None, None
    # locations_all = np.vstack(locations_all)    
    # index_ray_all = np.array(index_ray_all) 

    # tranform back
    locations_all[:,1] -= cad_lwh[2]/2.0

    # rotate x
    Rx = np.array([[1,0,0],[0,np.cos(np.pi/2.0),-np.sin(np.pi/2.0)],[0,np.sin(np.pi/2.0),np.cos(np.pi/2.0)]])
    locations_all = np.transpose(np.matmul(Rx, np.transpose(locations_all, (1,0))), (1,0))   

    # rotate z
    angle = bbox[6]
    Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    locations_all = np.transpose(np.matmul(Rz, np.transpose(locations_all, (1,0))), (1,0))    

    # translate
    locations_all += bbox[0:3]

    if 0:
        print(locations_all.shape[0])
        mesh.visual.face_colors = [200, 200, 250, 100]
        n = mesh.vertices.shape[0]
        pc3 = Points(locations_all, r=3, c='blue')   
        show([(mesh,pc3)], N=2, axes=1)          
        a = 0 

    return locations_all, index_ray_all  


def mesh_ray_tracing_mesh5(mesh, cad_lwh, ray_origin, pts, bbox=None, step_size=50, points=None, scale=None):           
    #mesh = as_mesh(mesh)
    #faces = mesh.faces
    #faces_area = mesh.area_faces
    #mesh.faces = faces[faces_area>0.001]
    
    ray_directions = pts - ray_origin
    ray_norm = np.linalg.norm(ray_directions, axis=1)
    ray_directions /= ray_norm[:,None]

    ray_directions[0,:] = ray_origin-bbox[0:3]
    # transform from car system to cad system
    # rotate z
    angle = -bbox[6]
    Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    ray_directions = np.transpose(np.matmul(Rz, np.transpose(ray_directions, (1,0))), (1,0))    

    # rotate x
    Rx = np.array([[1,0,0],[0,np.cos(-np.pi/2.0),-np.sin(-np.pi/2.0)],[0,np.sin(-np.pi/2.0),np.cos(-np.pi/2.0)]])
    ray_directions = np.transpose(np.matmul(Rx, np.transpose(ray_directions, (1,0))), (1,0))     
    
    # translate
    ray_directions[0,1] += cad_lwh[2]/2.0

    ray_origin_transformed = ray_directions[0,:]
    num_pts = ray_directions.shape[0]-1
    ray_origins = torch.tensor(ray_origin_transformed).cuda().float().view(1,-1).repeat(num_pts, 1)      
    ray_directions = torch.tensor(ray_directions[1:,:]).cuda().float()

    # intersection
    vertexes = torch.tensor(mesh.vertices).cuda().float()
    faces = torch.tensor(mesh.faces).cuda().int()
    ray_intersection_dis = mesh_ray_intersection_utils.mesh_ray_intersection_3d_cuda(vertexes, faces, ray_origins, ray_directions)

    # get intersection point
    ray_origins = ray_origins.cpu().numpy()
    ray_directions = ray_directions.cpu().numpy()
    ray_intersection_dis = ray_intersection_dis.cpu().numpy()
    locations_all = ray_origins+ray_directions*ray_intersection_dis
    index_ray_all = np.arange(0, ray_directions.shape[0], 1, dtype=int)
    mask_ray = ray_intersection_dis[:,0]<1000
    if not np.sum(mask_ray):
        return None, None
    locations_all = locations_all[mask_ray]
    index_ray_all = index_ray_all[mask_ray]

    # tranform back
    locations_all[:,1] -= cad_lwh[2]/2.0

    # rotate x
    Rx = np.array([[1,0,0],[0,np.cos(np.pi/2.0),-np.sin(np.pi/2.0)],[0,np.sin(np.pi/2.0),np.cos(np.pi/2.0)]])
    locations_all = np.transpose(np.matmul(Rx, np.transpose(locations_all, (1,0))), (1,0))   

    # rotate z
    angle = bbox[6]
    Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    locations_all = np.transpose(np.matmul(Rz, np.transpose(locations_all, (1,0))), (1,0))    
    if scale is not None:
        locations_all *= scale
        
    # translate
    locations_all += bbox[0:3]

    if 0:
        print(locations_all.shape[0])
        mesh.visual.face_colors = [200, 200, 250, 100]
        n = mesh.vertices.shape[0]
        pc1 = Points(pts, r=7, c='red')   
        pc3 = Points(locations_all, r=7, c='blue')   
        show([(pc1,pc3)], N=2, axes=1)          
        a = 0 

    return locations_all, index_ray_all  

