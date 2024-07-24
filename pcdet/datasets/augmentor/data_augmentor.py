from functools import partial
import numpy as np
import copy
import torch
from scipy.spatial import distance
from sklearn.cluster import KMeans

from . import augmentor_utils, database_sampler
from ...utils import common_utils, downsample_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        self.augmentor_configs = augmentor_configs

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)



    def get_polar_image(self, points):
        theta, phi = downsample_utils.compute_angles(points[:,:3])
        r = np.sqrt(np.sum(points[:,:3]**2, axis=1))
        polar_image = points.copy()
        polar_image[:,0] = phi 
        polar_image[:,1] = theta
        polar_image[:,2] = r 
        return polar_image

    def label_point_cloud_beam(self, polar_image, points, beam=32):
        if polar_image.shape[0] <= beam:
            print("too small point cloud!")
            return np.arange(polar_image.shape[0])
        beam_label, centroids = downsample_utils.beam_label(polar_image[:,1], beam)
        idx = np.argsort(centroids)
        rev_idx = np.zeros_like(idx)
        for i, t in enumerate(idx):
            rev_idx[t] = i
        beam_label = rev_idx[beam_label]
        return beam_label
        
    def random_beam_upsample(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_beam_upsample, config=config)
        points = data_dict['points']
        polar_image = self.get_polar_image(points)
        beam_label = self.label_point_cloud_beam(polar_image, points, config['BEAM'])
        new_pcs = [points]
        phi = polar_image[:,0]
        for i in range(config['BEAM'] - 1):
            if np.random.rand() < config['BEAM_PROB']:
                cur_beam_mask = (beam_label == i)
                next_beam_mask = (beam_label == i + 1)
                delta_phi = np.abs(phi[cur_beam_mask, np.newaxis] - phi[np.newaxis, next_beam_mask])
                corr_idx = np.argmin(delta_phi,1)
                min_delta = np.min(delta_phi,1)
                mask = min_delta < config['PHI_THRESHOLD']
                cur_beam = polar_image[cur_beam_mask][mask]
                next_beam = polar_image[next_beam_mask][corr_idx[mask]]
                new_beam = (cur_beam + next_beam)/2
                new_pc = new_beam.copy()
                new_pc[:,0] = np.cos(new_beam[:,1]) * np.cos(new_beam[:,0]) * new_beam[:,2]
                new_pc[:,1] = np.cos(new_beam[:,1]) * np.sin(new_beam[:,0]) * new_beam[:,2]
                new_pc[:,2] = np.sin(new_beam[:,1]) * new_beam[:,2]
                new_pcs.append(new_pc)
        data_dict['points'] = np.concatenate(new_pcs,0)
        return data_dict

    def random_beam_downsample(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_beam_downsample, config=config)
        points = data_dict['points']
        if 'beam_labels' in data_dict: # for waymo and kitti datasets
            beam_label = data_dict['beam_labels']
        else:
            polar_image = self.get_polar_image(points)
            beam_label = self.label_point_cloud_beam(polar_image, points, config['BEAM'])
        beam_mask = np.random.rand(config['BEAM']) < config['BEAM_PROB']
        points_mask = beam_mask[beam_label]
        data_dict['points'] = points[points_mask]
        if config.get('FILTER_GT_BOXES', None):
            num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
                        torch.from_numpy(data_dict['points'][:, :3]),
                        torch.from_numpy(data_dict['gt_boxes'][:, :7])).numpy().sum(axis=1)

            mask = (num_points_in_gt >= config.get('MIN_POINTS_OF_GT', 1))
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            data_dict['gt_names'] = data_dict['gt_names'][mask]
            if 'gt_classes' in data_dict:
                data_dict['gt_classes'] = data_dict['gt_classes'][mask]
                data_dict['gt_scores'] = data_dict['gt_scores'][mask]
            if 'gt_boxes_mask' in data_dict:
                data_dict['gt_boxes_mask'] = data_dict['gt_boxes_mask'][mask]

        return data_dict
    
    def remove_scans(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.remove_scans, config=config)
        
        # 
        points = data_dict['points']
        lxy = (points[:,0]**2+points[:,1]**2)**(0.5)
        points_altitude = np.arctan(points[:,2]/lxy).reshape(-1,1)
        random_idx = np.random.permutation(points_altitude.shape[0])
        points_altitude_selected = points_altitude[random_idx[0:int(0.1*points_altitude.shape[0])]]
        points_altitude_selected = np.sort(points_altitude_selected)
        kmeans = KMeans(n_clusters=config['RAYS_SOURCE'], random_state=0, n_init="auto").fit(points_altitude_selected)
        
        ray_ids = kmeans.predict(points_altitude)
        rays_kept = np.linspace(0, config['RAYS_SOURCE'], config['RAYS_TARGET']+1)
        rays_kept = rays_kept.astype(np.int64)
        mask = np.zeros(points_altitude.shape[0])
        for i in range(rays_kept.shape[0]):
            mask += ray_ids==rays_kept[i]
        points = points[mask>0]
        
        data_dict['points'] = points
        return data_dict
    

    def uniform_object_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.uniform_object_scaling, config=config)
        
        # calculate scale for each different object
        gt_boxes = data_dict['gt_boxes']
        mask = data_dict['gt_boxes_mask']
        idx_valid = np.where(mask>0)
        volume = gt_boxes[:,3]*gt_boxes[:,4]*gt_boxes[:,5]
        volume = volume[mask>0]
        volume_sort = np.sort(volume)
        idx_sort = np.argsort(volume)
        
        random_scales = np.random.uniform(config['SCALE_RANGE'][0], config['SCALE_RANGE'][1], volume.shape[0])
        volume_target = (random_scales**3)*config['SOURCE_VOLUME']
        volume_target = np.sort(volume_target)
        volume_scale = volume_target/volume_sort
        volume_scale = volume_scale**(1.0/3.0)

        scale_best = np.ones(gt_boxes.shape[0])    
        scale_best[idx_valid[0][idx_sort]] = volume_scale
        points, gt_boxes, gt_boxes_mask = augmentor_utils.scale_pre_object_with_scale(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            scale_list = scale_best
        )

        data_dict['gt_boxes_mask'] = gt_boxes_mask
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    # def self_geometrical_augmentation(self, data_dict=None, config=None):
    #     if data_dict is None:
    #         return partial(self.self_geometrical_augmentation, config=config)

    #     targte_good_sample_info = np.load(config['INFO_DB'])
    #     targte_good_sample_info[targte_good_sample_info[:,0]>1000,0] = 1000     
        
    #     # get out-box points      
    #     points = data_dict['points'][:,0:3]
    #     gt_boxes = data_dict['gt_boxes']

    #     box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
    #         torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
    #         torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
    #     ).long().squeeze(dim=0).cpu().numpy()    

    #     # simulate in-box points            
    #     points_inbox = []
    #     for i in range(gt_boxes.shape[0]):
    #         gt_points = points[box_idxs_of_pts == i]
    #         if gt_points.shape[0]>config['NUM_PTS_MAX'] or gt_points.shape[0]<5:
    #             points_inbox.append(gt_points)
    #             continue  

    #         x, y, z = gt_boxes[i,0], gt_boxes[i,1], gt_boxes[i,2]               
    #         l, w, h = gt_boxes[i,3], gt_boxes[i,4], gt_boxes[i,5]
    #         orientation = gt_boxes[i,6]
    #         direction = np.arctan2(y, x) # arctan2(y,x) in kitti-lidar-coordinate                
    #         alpha = (direction - orientation + np.pi)%(2*np.pi)   

    #         # normalized pts
    #         pts_n = gt_points[:,0:3]-gt_boxes[i,0:3]
    #         angle = -gt_boxes[i,6]
    #         Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    #         pts_n = np.transpose(np.matmul(Rz, np.transpose(pts_n, (1,0))), (1,0))  

    #         # find best fitted strong sample        
    #         dif_alpha = targte_good_sample_info[:,1]-alpha
    #         dif_alpha = np.abs(dif_alpha%(2*np.pi))  
    #         dif_size = np.abs(targte_good_sample_info[:,2]-l)+np.abs(targte_good_sample_info[:,3]-w)+np.abs(targte_good_sample_info[:,4]-h)
    #         match_dif = dif_size + 0.1*dif_alpha - 0.001*targte_good_sample_info[:,0]
    #         index_sort = np.argsort(match_dif)

    #         # algin points
    #         idx_select = index_sort[np.random.randint(0,10)]
    #         file_strong = config['FOLDER_DB']+str(idx_select)+'.npy'
    #         strong_points = np.load(file_strong)
    #         scale = gt_boxes[i,3:6]/targte_good_sample_info[idx_select,2:5]
    #         strong_points *= scale

    #         #dis_matrix = distance.cdist(pts_n, strong_points, 'euclidean')
    #         dis_matrix = torch.cdist(torch.tensor(pts_n).cuda().float(), torch.tensor(strong_points).cuda().float(), p=2)
    #         idx_min = torch.argmin(dis_matrix, dim=1).cpu()    
    #         pts_aug = strong_points[idx_min]    
    #         angle = gt_boxes[i,6]
    #         Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    #         pts_aug = np.transpose(np.matmul(Rz, np.transpose(pts_aug, (1,0))), (1,0))  
    #         pts_aug += gt_boxes[i,0:3]
    #         if pts_aug.shape[0]:
    #             points_inbox.append(pts_aug)

    #     if len(points_inbox):
    #         points_inbox = np.concatenate(points_inbox)
    #         points_outbox = points[box_idxs_of_pts < 0]
    #         points_final = np.concatenate([points_inbox, points_outbox])
    #     else:
    #         points_final = points                

    #     data_dict['points'] = points_final
    #     return data_dict


    def self_geometrical_augmentation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.self_geometrical_augmentation, config=config)

        targte_good_sample_info = np.load(config['INFO_DB'])
        targte_good_sample_info[targte_good_sample_info[:,0]>1000,0] = 1000     
        
        # get out-box points      
        points = data_dict['points'][:,0:3]
        gt_boxes = data_dict['gt_boxes']

        mask = np.zeros(points.shape[0])   
        pts_list = []         
        for j in range(gt_boxes.shape[0]):      
            box_expand = copy.deepcopy(gt_boxes[j:j+1])
            #box_expand[3:6] += 0.2
            mask_temp = roiaware_pool3d_utils.points_in_boxes_cpu(points, box_expand).squeeze(0)
            mask += mask_temp   
            pts_temp = points[mask_temp==1]
            pts_list.append(pts_temp)  
        points_outbox = points[mask==0]    

        # simulate in-box points            
        points_inbox = []
        for i in range(gt_boxes.shape[0]):            
            if pts_list[i].shape[0]>config['NUM_PTS_MAX'] or pts_list[i].shape[0]<5:
                points_inbox.append(pts_list[i])
                continue  

            x, y, z = gt_boxes[i,0], gt_boxes[i,1], gt_boxes[i,2]               
            l, w, h = gt_boxes[i,3], gt_boxes[i,4], gt_boxes[i,5]
            orientation = gt_boxes[i,6]
            direction = np.arctan2(y, x) # arctan2(y,x) in kitti-lidar-coordinate                
            alpha = (direction - orientation + np.pi)%(2*np.pi)   

            # normalized pts
            pts_n = pts_list[i][:,0:3]-gt_boxes[i,0:3]
            angle = -gt_boxes[i,6]
            Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            pts_n = np.transpose(np.matmul(Rz, np.transpose(pts_n, (1,0))), (1,0))  

            # find best fitted strong sample        
            dif_alpha = targte_good_sample_info[:,1]-alpha
            dif_alpha = np.abs(dif_alpha%(2*np.pi))  
            dif_size = np.abs(targte_good_sample_info[:,2]-l)+np.abs(targte_good_sample_info[:,3]-w)+np.abs(targte_good_sample_info[:,4]-h)
            match_dif = dif_size + 0.1*dif_alpha - 0.001*targte_good_sample_info[:,0]
            index_sort = np.argsort(match_dif)

            # algin points
            idx_select = index_sort[np.random.randint(0,5)]
            file_strong = config['FOLDER_DB']+str(idx_select)+'.npy'
            strong_points = np.load(file_strong)
            scale = gt_boxes[i,3:6]/targte_good_sample_info[idx_select,2:5]
            strong_points *= scale

            dis_matrix = distance.cdist(pts_n, strong_points, 'euclidean')
            #dis_matrix = torch.cdist(torch.tensor(pts_n).cuda().float(), torch.tensor(strong_points).cuda().float(), p=2)
            #idx_min = torch.argmin(dis_matrix, dim=1).cpu()    
            idx_min = np.argmin(dis_matrix, axis=1)
            pts_aug = strong_points[idx_min]    
            angle = gt_boxes[i,6]
            Rz = np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            pts_aug = np.transpose(np.matmul(Rz, np.transpose(pts_aug, (1,0))), (1,0))  
            pts_aug += gt_boxes[i,0:3]
            if pts_aug.shape[0]:
                points_inbox.append(pts_aug)

        if len(points_inbox):
            points_inbox = np.concatenate(points_inbox)
            points_final = np.concatenate([points_inbox, points_outbox])
        else:
            points_final = points                

        data_dict['points'] = points_final
        return data_dict
    

    def random_object_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_rotation, config=config)

        gt_boxes, points = augmentor_utils.rotate_objects(
            data_dict['gt_boxes'],
            data_dict['points'],
            data_dict['gt_boxes_mask'],
            rotation_perturb=config['ROT_UNIFORM_NOISE'],
            prob=config['ROT_PROB'],
            num_try=50
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_object_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_scaling, config=config)
        points, gt_boxes = augmentor_utils.scale_pre_object(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            scale_perturb=config['SCALE_UNIFORM_NOISE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_sampling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_sampling, config=config)
        gt_boxes, points, gt_boxes_mask = augmentor_utils.global_sampling(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            sample_ratio_range=config['WORLD_SAMPLE_RATIO'],
            prob=config['PROB']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_boxes_mask'] = gt_boxes_mask
        data_dict['points'] = points
        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def normalize_object_size(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.normalize_object_size, config=config)
        points, gt_boxes = augmentor_utils.normalize_object_size(
            data_dict['gt_boxes'], data_dict['points'], data_dict['gt_boxes_mask'], config['SIZE_RES']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict

    def re_prepare(self, augmentor_configs=None, intensity=None):
        self.data_augmentor_queue = []

        if augmentor_configs is None:
            augmentor_configs = self.augmentor_configs

        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            # scale data augmentation intensity
            if intensity is not None:
                cur_cfg = self.adjust_augment_intensity(cur_cfg, intensity)
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def adjust_augment_intensity(self, config, intensity):
        adjust_map = {
            'random_object_scaling': 'SCALE_UNIFORM_NOISE',
            'random_object_rotation': 'ROT_UNIFORM_NOISE',
            'random_world_rotation': 'WORLD_ROT_ANGLE',
            'random_world_scaling': 'WORLD_SCALE_RANGE',
        }

        def cal_new_intensity(config, flag):
            origin_intensity_list = config.get(adjust_map[config.NAME])
            assert len(origin_intensity_list) == 2
            assert np.isclose(flag - origin_intensity_list[0], origin_intensity_list[1] - flag)
            
            noise = origin_intensity_list[1] - flag
            new_noise = noise * intensity
            new_intensity_list = [flag - new_noise, new_noise + flag]
            return new_intensity_list

        if config.NAME not in adjust_map:
            return config
        
        # for data augmentations that init with 1
        if config.NAME in ["random_object_scaling", "random_world_scaling"]:
            new_intensity_list = cal_new_intensity(config, flag=1)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        elif config.NAME in ['random_object_rotation', 'random_world_rotation']:
            new_intensity_list = cal_new_intensity(config, flag=0)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        else:
            raise NotImplementedError