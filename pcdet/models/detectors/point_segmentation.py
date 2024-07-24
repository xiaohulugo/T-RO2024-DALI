import torch
import matplotlib.pyplot as plt
from .detector3d_template import Detector3DTemplate
from ..model_utils.model_nms_utils import class_agnostic_nms
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

import numpy as np
import mayavi.mlab as mlab

def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)
    
    if show_intensity:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        d = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)  # Map Distance from sensor
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], d, mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)
    mlab.show()
    return fig

class PointSegmentation(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        if batch_dict['domain'] == 'S':
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)
        else:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)

        if False:
            pts = batch_dict['point_coords'][:,1:4]
            visualize_pts(pts.detach().cpu().numpy())

        if True:
            pts = batch_dict['point_coords'][:,1:4]
            scores = batch_dict['point_cls_scores'].view(-1,1)
            pts_score = torch.cat([pts, 10*scores], dim=1)
            visualize_pts(pts_score.detach().cpu().numpy(), show_intensity=True)

        if self.training:
            if batch_dict['domain'] == 'S':
                loss, tb_dict, disp_dict = self.get_training_loss()
                ret_dict = {
                    'loss': loss
                }
            else:
                tb_dict = None
                disp_dict = None
                loss = batch_dict['ent_cls_point']
                ret_dict = {
                    'loss': loss
                }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        tb_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()

        loss = loss_point
        return loss, tb_dict, disp_dict
