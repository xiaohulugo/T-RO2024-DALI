from matplotlib.pyplot import axis
import _init_path
import os
import torch
import torch.nn as nn
import pickle
import random

from tensorboardX import SummaryWriter
import time
import glob
import re
import datetime
import argparse
import numpy as np
from pathlib import Path
import torch.distributed as dist
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from eval_utils import eval_utils
from visual_utils import visualize_utils
from train_utils.optimization import build_optimizer, build_scheduler
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.datasets.waymo.waymo_dataset import create_waymo_infos

from pcdet.datasets.augmentor.augmentor_utils import roiaware_pool3d_utils
from simulation_utils.mesh_ray_intersection import generate_rays, mesh_ray_tracing_mesh4, mesh_ray_tracing_mesh5
import trimesh
import json
from os.path import exists
import copy
import shutil

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=80, required=False, help='Number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()
    args.batch_size = 1
    args.workers = 0

    #args.cfg_file = '/home/lxh/Documents/Code/Detection3D/DALI_TRO/tools/cfgs/configs/secondiou_nuscenes2kitti.yaml'
    args.cfg_file = '/home/lxh/Documents/Code/Detection3D/DALI_TRO/tools/cfgs/configs/secondiou_waymo2kitti.yaml'
    #args.ckpt = '/home/lxh/Documents/Code/Detection3D/DALI_TRO/pretrained/CVPR2023/waymo_xyz_0.2portion/default/ckpt/checkpoint_epoch_10.pth'
    args.ckpt = '/home/lxh/Documents/Code/Detection3D/DALI_TRO/output/secondiou_waymo2kitti_iter1/default/ckpt/checkpoint_epoch_39.pth'
    args.scale = 1.25

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    #cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    cfg.EXP_GROUP_PATH = ''
    np.random.seed(1024)
    cfg.DATA_CONFIG_TAR.FOV_POINTS_ONLY = False

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()    

def as_mesh(scene):
    dump = scene.dump() 
    mesh_list = []   
    for mesh in dump:
        if mesh.faces.shape[1] !=3:
            continue
        mesh_list.append(mesh)
    
    mesh = trimesh.util.concatenate(tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)for g in mesh_list))
    return mesh


def RC_PPCG():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    ps_label_dir = output_dir / 'ps_label'
    ps_label_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------            
    # remember to disable data augmentor 
    cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR = None
    target_set, target_loader, target_sampler = build_dataloader(
        cfg.DATA_CONFIG_TAR, cfg.DATA_CONFIG_TAR.CLASS_NAMES, args.batch_size,
        dist_train, workers=args.workers, logger=logger, training=True,feat=cfg.CAD3D_CONFIG.FEATURE)

    # load model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.DATA_CONFIG_TAR.CLASS_NAMES), dataset=target_set)
    model.cuda()
    
    # load checkpoint if it is possible
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist)    
    model.eval()

    # read in cad car information
    folder_cad = cfg.CAD3D_CONFIG.CAD_PATH
    folder_out = cfg.CAD3D_CONFIG.RC_PPCG_PATH
    if exists(folder_out):
        shutil.rmtree(folder_out)
    os.mkdir(folder_out)
    os.mkdir(folder_out+'/label')
    os.mkdir(folder_out+'/lidar')
    
    with open(folder_cad+'/collection_clean.json') as json_file:
        cad_info = json.load(json_file)
    models_info = cad_info['models']
    models_lwh = []
    models_id = []
    models_mesh = []
    a = 0
    for i in range(len(models_info)):
        if models_info[i]['domain']=='emergency':
            continue
        if models_info[i]['type']=='trunk' or models_info[i]['type']=='bus':
            continue        
        if models_info[i]['length']>8 or models_info[i]['length']<2:
            continue 
        if models_info[i]['car_year'] is not None:
            if int(models_info[i]['car_year']) < 2000:
                continue                        
        id = models_info[i]['id']
        if exists(folder_cad+'/obj/'+id+'.obj'):
            l = models_info[i]['length']
            w = models_info[i]['width']
            h = models_info[i]['height']
            lwh = np.array([l,w,h])
            models_lwh.append(lwh.reshape(1,-1))
            models_id.append(id)
            mesh = trimesh.load(folder_cad+'/obj/'+id+'.obj')
            mesh = as_mesh(mesh)
            faces = mesh.faces
            a += faces.shape[0]
            models_mesh.append(mesh)
    models_lwh = np.concatenate(models_lwh).astype(float)
    models_id = np.array(models_id)
    print(a/models_lwh.shape[0])

    # # simulate rays
    if cfg.CAD3D_CONFIG.TARGET == 'nuscenes':
        nuscenes_cfg = {}
        nuscenes_cfg['origin'] = [0, 0, 1.8]
        nuscenes_cfg['v_fov'] = [-30.0,10.0]
        nuscenes_cfg['num_beam'] = 32
        nuscenes_cfg['beam_list'] = np.linspace(
            nuscenes_cfg['v_fov'][0]/180.0*np.pi, 
            nuscenes_cfg['v_fov'][1]/180.0*np.pi, 
            nuscenes_cfg['num_beam'])
        nuscenes_cfg['numpts_per_beam'] = 781 #781 1084
        nuscenes_cfg['angle_resolution_min'] = 360/nuscenes_cfg['numpts_per_beam']   
        simulation_cfg = nuscenes_cfg
    elif cfg.CAD3D_CONFIG.TARGET == 'kitti':
        kitti_cfg = {}
        kitti_cfg['origin'] = [0, 0, 1.6]
        kitti_cfg['v_fov'] = [-24.8,2.0]
        kitti_cfg['num_beam'] = 64
        kitti_cfg['beam_list'] = np.linspace(
            kitti_cfg['v_fov'][0]/180.0*np.pi, 
            kitti_cfg['v_fov'][1]/180.0*np.pi, 
            kitti_cfg['num_beam'])
        kitti_cfg['numpts_per_beam'] = 1863        
        kitti_cfg['angle_resolution_min'] = 360/kitti_cfg['numpts_per_beam']
        simulation_cfg = kitti_cfg
    else:
        print('Not implemented yet.')
        exit()
    
    ray_origins, ray_directions = generate_rays(simulation_cfg['origin'], 
                                                simulation_cfg['beam_list'],  
                                                simulation_cfg['numpts_per_beam'])
        
    # generate label based on prediction
    th_pts = cfg.CAD3D_CONFIG.TH_POINTS
    dataloader = target_loader
    dataloader.dataset.scale = args.scale
    dataloader_iter = iter(dataloader)
    label_save = []
    lidar_save = []
    name_save = []
    name_used = []
    count1 = 0
    count2 = 0
    with torch.no_grad():
        for i in range(len(dataloader)):
            try:
                batch_dict = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch_dict = next(dataloader_iter)
            name = batch_dict['frame_id'][0]
            postfix = '_'+str(int(100*dataloader.dataset.scale))
            print(name)
            if exists(folder_out+'/label/'+name+postfix+'.npy') or name in name_used:
                name_used.append(name)
                continue   

            load_data_to_gpu(batch_dict)   
            pred_dicts, ret_dict = model(batch_dict)
            pred_boxes = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
            pred_boxes[:,0:6] /= dataloader.dataset.scale
            pred_scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()  
            pred_boxes = pred_boxes[pred_scores>cfg.CAD3D_CONFIG.TH_SCORE]
            pred_scores = pred_scores[pred_scores>cfg.CAD3D_CONFIG.TH_SCORE]
            points = batch_dict['points'][:,1:4].detach().cpu().numpy()            
            points[:,0:3] /= dataloader.dataset.scale   
            name_used.append(name)

            # get out-box points                
            mask = np.zeros(points.shape[0])   
            pts_list = []             
            for k in range(pred_boxes.shape[0]):      
                box_expand = pred_boxes[k:k+1]
                box_expand[3:6] += 0.2
                mask_temp = roiaware_pool3d_utils.points_in_boxes_cpu(points, box_expand).squeeze(0)
                mask += mask_temp   
                pts_list.append(points[mask_temp==1])                 
            points_outbox = points[mask==0]  
            
            # simulate in-box points
            select_boxes = pred_boxes
            locations = []
            index_rays = []
            for j in range(select_boxes.shape[0]):   
                if pts_list[j].shape[0]<3:
                    continue  
                x, y, z = select_boxes[j,0], select_boxes[j,1], select_boxes[j,2]               
                l, w, h = select_boxes[j,3], select_boxes[j,4], select_boxes[j,5]
                lwh_ratio = copy.deepcopy(models_lwh)
                lwh_ratio[:,0] /= l
                lwh_ratio[:,1] /= w
                lwh_ratio[:,2] /= h
                diff = np.sum(np.abs(lwh_ratio-1.0), axis=1)
                idx_sort = np.argsort(diff)
                diff_sort = np.sort(diff)
                num_candidates = np.max([np.sum(diff_sort<0.3), 1])
                #aa = int(np.floor(np.random.rand(1)*num_candidates))
                aa = int(np.floor(np.random.rand(1)*5))
                best_idx = idx_sort[aa]
                scale_l = l/models_lwh[best_idx][0]
                scale_w = w/models_lwh[best_idx][1]
                scale_h = h/models_lwh[best_idx][2]
                scale = np.array([scale_l,scale_w,scale_h])
                
                print([i,j,num_candidates,int(models_id[best_idx])])

                if pts_list[j].shape[0] > th_pts:
                    locations.append(pts_list[j])    
                    count1 += 1
                else: 
                    #mesh = trimesh.load(folder_cad+'/obj/'+models_id[best_idx]+'.obj')
                    #mesh = as_mesh(mesh)
                    mesh = models_mesh[best_idx]
                    locations_j, _ = mesh_ray_tracing_mesh5(mesh, 
                                                        models_lwh[best_idx],
                                                        ray_origins[0,:],
                                                        pts_list[j], 
                                                        bbox=select_boxes[j,:])                 
                    if locations_j is None:
                        continue
                    locations.append(locations_j)
                    count2 += 1
            if len(locations):
                locations = np.concatenate(locations)
            else:                    
                # points[:,0:3] *= 1.0
                # select_boxes[:,0:6] *= 1.0                                  
                # np.save(folder_out+'/label/'+name+postfix+'.npy', select_boxes)
                # np.save(folder_out+'/lidar/'+name+postfix+'.npy', points)  
                if len(label_save)<50:
                    label_save.append(select_boxes)
                    lidar_save.append(points)
                    name_save.append(name)
                continue

            points_inbox = locations           
            points_final = np.concatenate([points_inbox, points_outbox])

            if False:
                visualize_utils.draw_scenes(points_final, select_boxes, batch_dict['gt_boxes'][0])
                visualize_utils.draw_scenes(points, pred_boxes, batch_dict['gt_boxes'][0])

            if len(label_save)<50:
                label_save.append(select_boxes)
                lidar_save.append(points_final)
                name_save.append(name)
            if len(label_save)==50 or i==len(dataloader)-1:
                for j in range(len(label_save)):
                    np.save(folder_out+'/label/'+name_save[j]+postfix+'.npy', label_save[j])
                    np.save(folder_out+'/lidar/'+name_save[j]+postfix+'.npy', lidar_save[j]) 
                label_save = []
                lidar_save = []
                name_save = []

            # points_final[:,0:3] *= 1.0
            # select_boxes[:,0:6] *= 1.0                
            # np.save(folder_out+'/label/'+name+postfix+'.npy', select_boxes)
            # np.save(folder_out+'/lidar/'+name+postfix+'.npy', points_final)         
    print([count1, count2])

    
def CF_PPCG():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    ps_label_dir = output_dir / 'ps_label'
    ps_label_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------            
    # remember to disable data augmentor 
    cfg.DATA_CONFIG_TAR.DATA_AUGMENTOR = None
    target_set, target_loader, target_sampler = build_dataloader(
        cfg.DATA_CONFIG_TAR, cfg.DATA_CONFIG_TAR.CLASS_NAMES, args.batch_size,
        dist_train, workers=args.workers, logger=logger, training=True,feat=cfg.CAD3D_CONFIG.FEATURE)

    # load model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.DATA_CONFIG_TAR.CLASS_NAMES), dataset=target_set)
    model.cuda()
    
    # load checkpoint if it is possible
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist)    
    model.eval()

    # read in cad car information
    folder_cad = cfg.CAD3D_CONFIG.CAD_PATH
    folder_out = cfg.CAD3D_CONFIG.CF_PPCG_PATH
    if exists(folder_out):
        shutil.rmtree(folder_out)
    os.mkdir(folder_out)
    os.mkdir(folder_out+'/label')
    os.mkdir(folder_out+'/lidar')
    
    
    with open(folder_cad+'/collection_clean.json') as json_file:
        cad_info = json.load(json_file)
    models_info = cad_info['models']
    models_lwh = []
    models_id = []
    models_mesh = []
    for i in range(len(models_info)):
        if models_info[i]['domain']=='emergency':
            continue
        if models_info[i]['type']=='trunk' or models_info[i]['type']=='bus':
            continue        
        if models_info[i]['length']>8 or models_info[i]['length']<2:
            continue 
        if models_info[i]['car_year'] is not None:
            if int(models_info[i]['car_year']) < 2000:
                continue                        
        id = models_info[i]['id']
        if exists(folder_cad+'/obj/'+id+'.obj'):
            l = models_info[i]['length']
            w = models_info[i]['width']
            h = models_info[i]['height']
            lwh = np.array([l,w,h])
            models_lwh.append(lwh.reshape(1,-1))
            models_id.append(id)
            mesh = trimesh.load(folder_cad+'/obj/'+id+'.obj')
            mesh = as_mesh(mesh)
            models_mesh.append(mesh)
    models_lwh = np.concatenate(models_lwh).astype(float)
    models_id = np.array(models_id)

    if cfg.CAD3D_CONFIG.TARGET == 'nuscenes':
        nuscenes_cfg = {}
        nuscenes_cfg['origin'] = [0, 0, 1.8]
        nuscenes_cfg['v_fov'] = [-30.0,10.0]
        nuscenes_cfg['num_beam'] = 32
        nuscenes_cfg['beam_list'] = np.linspace(
            nuscenes_cfg['v_fov'][0]/180.0*np.pi, 
            nuscenes_cfg['v_fov'][1]/180.0*np.pi, 
            nuscenes_cfg['num_beam'])
        nuscenes_cfg['numpts_per_beam'] = 781 #781 1084
        nuscenes_cfg['angle_resolution_min'] = 360/nuscenes_cfg['numpts_per_beam']   
        simulation_cfg = nuscenes_cfg
    elif cfg.CAD3D_CONFIG.TARGET == 'kitti':
        kitti_cfg = {}
        kitti_cfg['origin'] = [0, 0, 1.6]
        kitti_cfg['v_fov'] = [-24.8,2.0]
        kitti_cfg['num_beam'] = 64
        kitti_cfg['beam_list'] = np.linspace(
            kitti_cfg['v_fov'][0]/180.0*np.pi, 
            kitti_cfg['v_fov'][1]/180.0*np.pi, 
            kitti_cfg['num_beam'])
        kitti_cfg['numpts_per_beam'] = 1863        
        kitti_cfg['angle_resolution_min'] = 360/kitti_cfg['numpts_per_beam']
        simulation_cfg = kitti_cfg
    else:
        print('Not implemented yet.')
        exit()
    
    
    ray_origins, ray_directions = generate_rays(simulation_cfg['origin'], 
                                                simulation_cfg['beam_list'],  
                                                simulation_cfg['numpts_per_beam'])
        
    # generate label based on prediction
    dataloader = target_loader
    dataloader.dataset.scale = args.scale
    dataloader_iter = iter(dataloader)
    label_save = []
    lidar_save = []
    name_save = []
    name_used = []    
    save_batch = 50
    with torch.no_grad():
        for i in range(len(dataloader)):
            try:
                batch_dict = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch_dict = next(dataloader_iter)

            name = batch_dict['frame_id'][0]
            postfix = '_'+str(int(100*dataloader.dataset.scale))
            print(name)
            if exists(folder_out+'/label/'+name+postfix+'.npy') or name in name_used:
                name_used.append(name)
                continue   

            load_data_to_gpu(batch_dict)   
            pred_dicts, ret_dict = model(batch_dict)
            pred_boxes = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
            pred_boxes[:,0:6] /= dataloader.dataset.scale
            pred_scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()  
            pred_boxes = pred_boxes[pred_scores>cfg.CAD3D_CONFIG.TH_SCORE]
            pred_scores = pred_scores[pred_scores>cfg.CAD3D_CONFIG.TH_SCORE]
            points = batch_dict['points'][:,1:4].detach().cpu().numpy()            
            points[:,0:3] /= dataloader.dataset.scale   
            name_used.append(name)

            # generate points and labels
            #id = batch_dict['sample_idx'][0]
                       
            # get out-box points                
            mask = np.zeros(points.shape[0])                
            for k in range(pred_boxes.shape[0]):      
                box_expand = pred_boxes[k:k+1]
                box_expand[3:6] += 0.2
                mask_temp = roiaware_pool3d_utils.points_in_boxes_cpu(points, box_expand).squeeze(0)
                mask += mask_temp                    
            points_outbox = points[mask==0]  
            
            # simulate in-box points
            select_boxes = pred_boxes
            locations = []
            index_rays = []
            for j in range(select_boxes.shape[0]):     
                x, y, z = select_boxes[j,0], select_boxes[j,1], select_boxes[j,2]               
                l, w, h = select_boxes[j,3], select_boxes[j,4], select_boxes[j,5]
                lwh_ratio = copy.deepcopy(models_lwh)
                lwh_ratio[:,0] /= l
                lwh_ratio[:,1] /= w
                lwh_ratio[:,2] /= h
                diff = np.sum(np.abs(lwh_ratio-1.0), axis=1)
                idx_sort = np.argsort(diff)
                diff_sort = np.sort(diff)
                num_candidates = np.max([np.sum(diff_sort<0.3), 1])
                #aa = int(np.floor(np.random.rand(1)*num_candidates))
                aa = int(np.floor(np.random.rand(1)*5))
                best_idx = idx_sort[aa]
                scale_l = l/models_lwh[best_idx][0]
                scale_w = w/models_lwh[best_idx][1]
                scale_h = h/models_lwh[best_idx][2]
                scale = np.array([scale_l,scale_w,scale_h])
                
                print([i,j,num_candidates,int(models_id[best_idx])])
                shift_origin = np.array([0,0,0])                 
                dis = (x**2+y**2)**(0.5)   
                dd = random.randint(20, 50)                                        
                t = dd/(dis+0.001)-1.0
                shift_origin = np.array([t*x,t*y,0])
                ray_origins_shifted = ray_origins[0,:]-shift_origin
                
                #mesh = trimesh.load(folder_cad+'/obj/'+models_id[best_idx]+'.obj')
                #mesh = as_mesh(mesh)
                mesh = models_mesh[best_idx]
                locations_j, index_rays_j = mesh_ray_tracing_mesh4(mesh, 
                                                    models_lwh[best_idx],
                                                    ray_origins_shifted,
                                                    ray_directions, 
                                                    bbox=select_boxes[j,:])
                
                if locations_j is None:
                    continue
                locations.append(locations_j)
                index_rays.append(index_rays_j)
            if len(locations):
                locations = np.concatenate(locations)
                index_rays = np.concatenate(index_rays)
            else:
                # points[:,0:3] *= dataloader.dataset.scale
                # select_boxes[:,0:6] *= dataloader.dataset.scale                     
                # np.save(folder_out+'/label/'+name+postfix+'.npy', pred_boxes)
                # np.save(folder_out+'/lidar/'+name+postfix+'.npy', points)  
                # continue
                if len(label_save)<save_batch:
                    label_save.append(select_boxes)
                    lidar_save.append(points)
                    name_save.append(name)
                continue

            # choose closest pt for ray
            best_dis = np.ones(ray_origins.shape[0])*1000000.0
            best_ptid = -np.ones(ray_origins.shape[0]).astype(int)
            for j in range(locations.shape[0]):
                id_ray = index_rays[j]
                dx = locations[j,0]-simulation_cfg['origin'][0]
                dy = locations[j,1]-simulation_cfg['origin'][1]
                dz = locations[j,2]-simulation_cfg['origin'][2]
                dis = dx*dx+dy*dy+dz*dz
                if dis<best_dis[id_ray]:
                    best_dis[id_ray] = dis
                    best_ptid[id_ray] = j
            ptid_selected = best_ptid[best_ptid>=0]
            points_inbox = locations[ptid_selected]            
            points_final = np.concatenate([points_inbox, points_outbox])

            if False:
                visualize_utils.draw_scenes(points_final, select_boxes, batch_dict['gt_boxes'][0])
                visualize_utils.draw_scenes(points, pred_boxes, batch_dict['gt_boxes'][0])

            if len(label_save)<save_batch:
                label_save.append(select_boxes)
                lidar_save.append(points_final)
                name_save.append(name)
            if len(label_save)==save_batch or i==len(dataloader)-1:
                for j in range(len(label_save)):
                    np.save(folder_out+'/label/'+name_save[j]+postfix+'.npy', label_save[j])
                    np.save(folder_out+'/lidar/'+name_save[j]+postfix+'.npy', lidar_save[j]) 
                label_save = []
                lidar_save = []
                name_save = []

            # points_final[:,0:3] *= dataloader.dataset.scale
            # select_boxes[:,0:6] *= dataloader.dataset.scale                
            # np.save(folder_out+'/label/'+name+postfix+'.npy', select_boxes)
            # np.save(folder_out+'/lidar/'+name+postfix+'.npy', points_final)         

if __name__ == '__main__':
    RC_PPCG()
    CF_PPCG()
