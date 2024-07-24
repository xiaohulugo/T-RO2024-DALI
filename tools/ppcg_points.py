from itertools import count
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
from simulation_utils.mesh_ray_intersection import generate_rays, mesh_ray_tracing_point1, mesh_ray_tracing_point2
import trimesh
import json
from os.path import exists
import copy
import shutil
from scipy.spatial import distance

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
    args.cfg_file = '/home/lxh/Documents/Code/Detection3D/DALI_TRO/tools/cfgs/configs/secondiou_waymo2kitti_points.yaml'
    #args.ckpt = '/home/lxh/Documents/Code/Detection3D/DALI_TRO/pretrained/CVPR2023/waymo_xyz_0.2portion/default/ckpt/checkpoint_epoch_10.pth'
    args.ckpt = '/home/lxh/Documents/Code/Detection3D/DALI_TRO/output/secondiou_waymo2kitti_points/default/ckpt/checkpoint_epoch_40.pth'
    args.scale = 1.20

    # w2n: 1.06(21.8), 1.00(25.1), 1.00(25.1)
    # w2n: 1.06(24.6), 1.00(26.0), 1.00() 
    # w2n: 1.06(18.4), 1.03(19.4), 1.03(21.3) 

    # n2k: 1.25(59.4), 1.10(64.7), 1.05(self 67.7)
    # n2k: 1.25(69.3), 1.10(75.2), 
    # n2k: 1.20(34.3), 1.15(43.6), 1.10(45.5), 1.05(self 56.1)

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



def generate_source_db():
    args, cfg = parse_config()
    folder_out = cfg.CAD3D_CONFIG.SOURCE_DA_PATH
    if not exists(folder_out):
        os.mkdir(folder_out)
        os.mkdir(folder_out+'/objects')

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    #cfg.DATA_CONFIG.SAMPLED_INTERVAL.train = 5    
    cfg.DATA_CONFIG.DATA_AUGMENTOR = None
    source_set, source_loader, source_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.DATA_CONFIG.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=args.epochs,
        feat = cfg.CAD3D_CONFIG.FEATURE)

    # get source db
    dataloader = source_loader
    th_numpts = 300
    th_sample = np.min([len(dataloader), 10000])
    
    # generate label based on prediction
    dataloader_iter = iter(dataloader)
    count = 0
    dbinfo = []
    with torch.no_grad():
        for i in range(th_sample):
            try:
                batch_dict = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch_dict = next(dataloader_iter)

            # generate points and labels
            name = batch_dict['frame_id'][0]
            #print(name)
            if i%100 == 0:
                print(i)

            gt_boxes = batch_dict['gt_boxes'][0]
            points = batch_dict['points'][:,1:4]
            for i in range(gt_boxes.shape[0]):
                box_i = gt_boxes[i:i+1,0:7]
                direction_i = np.arctan2(box_i[0,1], box_i[0,0])
                orientation_i = box_i[0,6]
                size_i = (box_i[0,0]*box_i[0,1]*box_i[0,2])**(1.0/3.0)

                point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:,0:3], box_i).squeeze(0)
                points_i = points[point_masks==1]
                if points_i.shape[0]<th_numpts:
                    continue       
                points_i[:,0:3] -= box_i[0,0:3]
                angle = -box_i[0,6]
                Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
                points_i[:,0:3] = np.transpose(np.matmul(Rz, np.transpose(points_i[:,0:3], (1,0))), (1,0))  
                
                #visualize_utils.draw_scenes(points)
                # pc = Points(points_i, r=3, c='black')            
                # show([(pc)], N=2, axes=1)          
                # a = 0                 

                info = {
                    'box':box_i,
                    'direction': direction_i,
                    'orientation': orientation_i,
                    'score': 1.0,
                    'numpts': points_i.shape[0],
                    'size': size_i,
                    'idx': count,
                    'name': name}
                dbinfo.append(info)
                np.save(folder_out + '/objects/' + str(count)+'.npy', points_i)            
                count+=1   
    print(count)
    with open(folder_out+'/dbinfo.pkl', 'wb') as f:
        pickle.dump(dbinfo, f)   

    return 0 


def RC_PPCG():
    args, cfg = parse_config()

    folder_source = cfg.CAD3D_CONFIG.SOURCE_DA_PATH
    folder_out = cfg.CAD3D_CONFIG.RC_PPCG_PATH
    target = cfg.CAD3D_CONFIG.TARGET

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

    # read in source object information
    #folder_source = '/media/lxh/Data/Aug/Waymo'
    with open(folder_source+'/dbinfo.pkl', 'rb') as f:
        source_db = pickle.load(f)   
    source_info = np.array([[x['score'], x['numpts'], x['box'][0,3], x['box'][0,4], x['box'][0,5], x['box'][0,6], x['idx'], x['direction']] for x in source_db])
    source_info = source_info[source_info[:,1]>300]
    source_points_list = []
    for i in range(source_info.shape[0]):
        info_i = source_info[i,:]
        ref_points = np.load(folder_source + '/objects/' + str(int(info_i[6]))+'.npy')  
        source_points_list.append(ref_points)      

    if exists(folder_out):
        shutil.rmtree(folder_out)
    os.mkdir(folder_out)
    os.mkdir(folder_out+'/label')
    os.mkdir(folder_out+'/lidar')

    th_score = cfg.CAD3D_CONFIG.TH_SCORE
    if target == 'kitti':
        ray_origins = np.array([0.0,0.0,1.6])
    else:
        ray_origins = np.array([0.0,0.0,1.8])

    # generate label based on prediction
    th_pts = cfg.CAD3D_CONFIG.TH_POINTS
    count1 = 0
    count2 = 0
    dataloader = target_loader
    dataloader.dataset.scale = args.scale
    dataloader_iter = iter(dataloader)    
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
            if exists(folder_out+'/label/'+name+postfix+'.npy'):
                continue   

            load_data_to_gpu(batch_dict)   
            pred_dicts, ret_dict = model(batch_dict)
            pred_boxes = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
            pred_boxes[:,0:6] /= dataloader.dataset.scale
            pred_scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()  
            pred_boxes = pred_boxes[pred_scores>th_score]
            pred_scores = pred_scores[pred_scores>th_score]
            points = batch_dict['points'][:,1:4].detach().cpu().numpy()            
            points[:,0:3] /= dataloader.dataset.scale   

            # get real in-box points                
            mask = np.zeros(points.shape[0])   
            pts_list = []         
            select_boxes = []    
            for k in range(pred_boxes.shape[0]):      
                box_expand = copy.deepcopy(pred_boxes[k:k+1])
                #box_expand[3:6] += 0.2
                mask_temp = roiaware_pool3d_utils.points_in_boxes_cpu(points, box_expand).squeeze(0)
                mask += mask_temp   
                pts_temp = points[mask_temp==1]
                if pts_temp.shape[0]<10:
                    continue
                pts_list.append(pts_temp)  
                select_boxes.append(pred_boxes[k:k+1])               
            points_outbox = points[mask==0]              
            if not len(select_boxes):
                continue
            select_boxes = np.vstack(select_boxes)
            
            # simulate in-box points
            aug_inbox_points = []
            for j in range(select_boxes.shape[0]):   
                if pts_list[j].shape[0] < 3:
                    continue  
                if pts_list[j].shape[0] > th_pts:
                    aug_inbox_points.append(pts_list[j])    
                    count1 += 1
                    continue

                x, y, z = select_boxes[j,0], select_boxes[j,1], select_boxes[j,2]               
                l, w, h = select_boxes[j,3], select_boxes[j,4], select_boxes[j,5]
                ori = select_boxes[j,6]
                dir = np.arctan2(y, x)

                points_ = pts_list[j][:,0:3] - select_boxes[j,0:3]
                angle = -ori
                Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
                points_[:,0:3] = np.transpose(np.matmul(Rz, np.transpose(points_[:,0:3], (1,0))), (1,0))  
                if points_.shape[0]>200:
                    tt = np.random.permutation(points_.shape[0])
                    points_ = points_[tt[0:200]]

                # find the best ref-box
                dif_dir = source_info[:,7]-dir
                dif_dir = dif_dir%(2*np.pi)
                dif_dir = np.abs(dif_dir)
                dif_ori = source_info[:,5]-ori
                dif_ori = dif_ori%(2*np.pi)
                dif_ori = np.abs(dif_ori)

                dif_total = dif_dir+dif_ori            
                index_sort = np.argsort(dif_total)
                index_select = index_sort[0:20]
                dis_list = []
                for k in range(len(index_select)):
                    ref_points = source_points_list[index_select[k]]
                    dis_matrix = distance.cdist(points_, ref_points[:,0:3], 'euclidean')
                    dis = np.sum(np.min(dis_matrix, axis=1))
                    dis_list.append(dis)             
                tt = np.argmin(np.array(dis_list))
                index_best = index_select[tt]

                # pts_count = source_info[index_select,1]
                # index_sort2 = np.argsort(pts_count)
                # aa = int(np.floor(np.random.rand(1)*10))
                # index_best = index_select[index_sort2[aa]]

                scale_l = l/source_info[index_best,2]
                scale_w = w/source_info[index_best,3]
                scale_h = h/source_info[index_best,4]
                print([i,j,pts_list[j].shape[0]])

                points_ref = copy.deepcopy(source_points_list[index_best])
                points_ref[:,0] *= scale_l
                points_ref[:,1] *= scale_w
                points_ref[:,2] *= scale_h
                points_aug_temp, _ = mesh_ray_tracing_point1(points_ref, 
                                                    ray_origins,
                                                    pts_list[j], 
                                                    bbox=select_boxes[j,:])                 
                if points_aug_temp is None:
                    aug_inbox_points.append(pts_list[j])
                else:
                    aug_inbox_points.append(points_aug_temp)
                count2 += 1
            if len(aug_inbox_points):
                aug_inbox_points = np.concatenate(aug_inbox_points)
                points_final = np.concatenate([aug_inbox_points, points_outbox])
            else:
                points_final = points_outbox

            if False:
                visualize_utils.draw_points(points, points_final)
                visualize_utils.draw_scenes(points_final, batch_dict['gt_boxes'][0],select_boxes)
                visualize_utils.draw_scenes(points, batch_dict['gt_boxes'][0], select_boxes)
          
            np.save(folder_out+'/label/'+name+postfix+'.npy', select_boxes)
            np.save(folder_out+'/lidar/'+name+postfix+'.npy', points_final)         
    print([count1, count2])



def CF_PPCG():
    args, cfg = parse_config()

    folder_source = cfg.CAD3D_CONFIG.SOURCE_DA_PATH
    folder_out = cfg.CAD3D_CONFIG.CF_PPCG_PATH
    target = cfg.CAD3D_CONFIG.TARGET

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

    # read in source object information
    #folder_source = '/media/lxh/Data/Aug/Waymo'
    with open(folder_source+'/dbinfo.pkl', 'rb') as f:
        source_db = pickle.load(f)   
    source_info = np.array([[x['score'], x['numpts'], x['box'][0,3], x['box'][0,4], x['box'][0,5], x['box'][0,6], x['idx'], x['direction']] for x in source_db])
    source_info = source_info[source_info[:,1]>300]
    source_points_list = []
    for i in range(source_info.shape[0]):
        info_i = source_info[i,:]
        ref_points = np.load(folder_source + '/objects/' + str(int(info_i[6]))+'.npy')  
        source_points_list.append(ref_points)      

    if exists(folder_out):
        shutil.rmtree(folder_out)
    os.mkdir(folder_out)
    os.mkdir(folder_out+'/label')
    os.mkdir(folder_out+'/lidar')

    th_score = cfg.CAD3D_CONFIG.TH_SCORE
    if target == 'kitti':
        ray_origins = np.array([0.0,0.0,1.6])
        num_beam = 64
        numpts_per_beam = 1800
    else:
        ray_origins = np.array([0.0,0.0,1.8])
        num_beam = 32
        numpts_per_beam = 780

    # # simulate rays
    simulation_cfg = {}
    simulation_cfg['origin'] = ray_origins #1.8 nuscenes, 1.6 kitti
    simulation_cfg['v_fov'] = [-27.4,6.0]
    simulation_cfg['num_beam'] = num_beam
    simulation_cfg['beam_list'] = np.linspace(
        simulation_cfg['v_fov'][0]/180.0*np.pi, 
        simulation_cfg['v_fov'][1]/180.0*np.pi, 
        simulation_cfg['num_beam'])
    simulation_cfg['numpts_per_beam'] = numpts_per_beam
    simulation_cfg['angle_resolution_min'] = 360/simulation_cfg['numpts_per_beam']   
    
    ray_origins, ray_directions = generate_rays(simulation_cfg['origin'], 
                                                simulation_cfg['beam_list'],  
                                                simulation_cfg['numpts_per_beam'])
    
    # generate label based on prediction
    th_pts = cfg.CAD3D_CONFIG.TH_POINTS
    count1 = 0
    count2 = 0
    dataloader = target_loader
    dataloader.dataset.scale = args.scale
    dataloader_iter = iter(dataloader)    
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
            if exists(folder_out+'/label/'+name+postfix+'.npy'):
                continue   

            load_data_to_gpu(batch_dict)   
            pred_dicts, ret_dict = model(batch_dict)
            pred_boxes = pred_dicts[0]['pred_boxes'].detach().cpu().numpy()
            pred_boxes[:,0:6] /= dataloader.dataset.scale
            pred_scores = pred_dicts[0]['pred_scores'].detach().cpu().numpy()  
            pred_boxes = pred_boxes[pred_scores>th_score]
            pred_scores = pred_scores[pred_scores>th_score]
            points = batch_dict['points'][:,1:4].detach().cpu().numpy()            
            points[:,0:3] /= dataloader.dataset.scale   

            # get real in-box points                
            mask = np.zeros(points.shape[0])   
            pts_list = []         
            select_boxes = []    
            for k in range(pred_boxes.shape[0]):      
                box_expand = copy.deepcopy(pred_boxes[k:k+1])
                #box_expand[3:6] += 0.2
                mask_temp = roiaware_pool3d_utils.points_in_boxes_cpu(points, box_expand).squeeze(0)
                mask += mask_temp   
                pts_temp = points[mask_temp==1]
                if pts_temp.shape[0]<5:
                    continue
                pts_list.append(pts_temp)  
                select_boxes.append(pred_boxes[k:k+1])               
            points_outbox = points[mask==0]              
            if not len(select_boxes):
                continue
            select_boxes = np.vstack(select_boxes)
            
            # simulate in-box points
            aug_inbox_points = []
            aug_index_rays = []
            for j in range(select_boxes.shape[0]):   
                if pts_list[j].shape[0] < 3:
                    continue  

                x, y, z = select_boxes[j,0], select_boxes[j,1], select_boxes[j,2]               
                l, w, h = select_boxes[j,3], select_boxes[j,4], select_boxes[j,5]
                ori = select_boxes[j,6]
                dir = np.arctan2(y, x)

                points_ = pts_list[j][:,0:3] - select_boxes[j,0:3]
                angle = -ori
                Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
                points_[:,0:3] = np.transpose(np.matmul(Rz, np.transpose(points_[:,0:3], (1,0))), (1,0))  
                if points_.shape[0]>200:
                    tt = np.random.permutation(points_.shape[0])
                    points_ = points_[tt[0:200]]

                # find the best ref-box
                dif_dir = source_info[:,7]-dir
                dif_dir = dif_dir%(2*np.pi)
                dif_dir = np.abs(dif_dir)
                dif_ori = source_info[:,5]-ori
                dif_ori = dif_ori%(2*np.pi)
                dif_ori = np.abs(dif_ori)

                dif_total = dif_dir+dif_ori            
                index_sort = np.argsort(dif_total)
                index_select = index_sort[0:20]
                dis_list = []
                for k in range(len(index_select)):
                    ref_points = source_points_list[index_select[k]]
                    dis_matrix = distance.cdist(points_, ref_points[:,0:3], 'euclidean')
                    dis = np.sum(np.min(dis_matrix, axis=1))
                    dis_list.append(dis)             
                tt = np.argmin(np.array(dis_list))
                index_best = index_select[tt]

                # pts_count = source_info[index_select,1]
                # index_sort2 = np.argsort(pts_count)
                # aa = int(np.floor(np.random.rand(1)*10))
                # index_best = index_select[index_sort2[aa]]

                scale_l = l/source_info[index_best,2]
                scale_w = w/source_info[index_best,3]
                scale_h = h/source_info[index_best,4]
                print([i,j,pts_list[j].shape[0]])

                dis = (x**2+y**2)**(0.5)   
                dd = random.randint(40, 60)                                        
                t = dd/(dis+0.001)-1.0
                shift = np.array([t*x,t*y,0])
                
                points_ref = copy.deepcopy(source_points_list[index_best])
                points_ref[:,0] *= scale_l
                points_ref[:,1] *= scale_w
                points_ref[:,2] *= scale_h
                points_aug_temp, index_rays_temp = mesh_ray_tracing_point2(points_ref, 
                                                    ray_origins[0,:],
                                                    shift,
                                                    ray_directions, 
                                                    bbox=select_boxes[j,:])
                                             
                if points_aug_temp is None:
                    continue
                else:
                    aug_inbox_points.append(points_aug_temp)
                    aug_index_rays.append(index_rays_temp)
                count2 += 1
            if len(aug_inbox_points):
                aug_inbox_points = np.concatenate(aug_inbox_points)
                aug_index_rays = np.concatenate(aug_index_rays)
            else:
                np.save(folder_out+'/label/'+name+postfix+'.npy', select_boxes)
                np.save(folder_out+'/lidar/'+name+postfix+'.npy', points)  
                continue   
                
            # blocking analysis
            best_dis = np.ones(ray_origins.shape[0])*1000000.0
            best_ptid = -np.ones(ray_origins.shape[0]).astype(int)
            for j in range(aug_inbox_points.shape[0]):
                id_ray = aug_index_rays[j]
                dx = aug_inbox_points[j,0]-simulation_cfg['origin'][0]
                dy = aug_inbox_points[j,1]-simulation_cfg['origin'][1]
                dz = aug_inbox_points[j,2]-simulation_cfg['origin'][2]
                dis = dx*dx+dy*dy+dz*dz
                if dis<best_dis[id_ray]:
                    best_dis[id_ray] = dis
                    best_ptid[id_ray] = j
            ptid_selected = best_ptid[best_ptid>=0]
            points_inbox = aug_inbox_points[ptid_selected]            
            points_final = np.concatenate([points_inbox, points_outbox])

            if False:
                visualize_utils.draw_points(points, points_final)
                visualize_utils.draw_scenes(points_final, batch_dict['gt_boxes'][0],select_boxes)
                visualize_utils.draw_scenes(points, batch_dict['gt_boxes'][0], select_boxes)
          
            np.save(folder_out+'/label/'+name+postfix+'.npy', select_boxes)
            np.save(folder_out+'/lidar/'+name+postfix+'.npy', points_final)         
    print([count1, count2])


if __name__ == '__main__':
    if False:
        gt = np.load('/media/lxh/Data/Aug/test/cf/label/000034_130.npy')
        points = np.load('/media/lxh/Data/Aug/test/cf/lidar/000034_130.npy')
        visualize_utils.draw_scenes(points, gt) 
        a = 0 
    if False:
        get_points_from_mesh()    

    #generate_source_db()
    RC_PPCG()
    CF_PPCG()
