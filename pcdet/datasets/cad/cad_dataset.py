import copy
import pickle
import glob
import os
import numpy as np
from os import listdir
from os.path import exists,isfile
from skimage import io
from ..dataset import DatasetTemplate

class CADDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None,feat='xyz'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger,feat=feat
        )
        self.path_list = dataset_cfg.PATH
        self.points_list = []
        self.labels_list = []        
        for path in self.path_list:
            #names = [f for f in listdir(path+'/label') if isfile(path+'/label/'+f)]
            names = [os.path.basename(f) for f in glob.glob(path+'/label/*.npy')]  
            self.points_list += [path+'/lidar/'+f for f in names]
            self.labels_list += [path+'/label/'+f for f in names]
        
        #self.samples = [os.path.basename(f) for f in glob.glob(self.path+'/label/*ori.npy')]  
        print('CAD samples: '+str(len(self.labels_list)))
        a = 0

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):     
        index = index % len(self.labels_list) 
        frame_id = index
        points = np.load(self.points_list[index])
        gt_boxes = np.load(self.labels_list[index])
        gt_names = np.array([self.class_names[0] for i in range(gt_boxes.shape[0])])
        if True:
            points[:,0:3] *= self.scale
            gt_boxes[:,0:6] *= self.scale
                
        input_dict = {
            'points': points,
            'gt_boxes':gt_boxes[:,0:7],
            'gt_names':gt_names,
            'frame_id': frame_id,
            'calib': None,
            'image_shape': 0
        }

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict