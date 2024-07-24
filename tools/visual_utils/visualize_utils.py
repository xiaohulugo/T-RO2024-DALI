"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import open3d
import torch
import matplotlib
import numpy as np
from matplotlib import cm

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 2.0    
    vis.get_render_option().background_color = np.ones(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if points.shape[1]==4:
        points[:,3][points[:,3]<0.6] = 0.0
        num_colors = 10
        color_map = cm.get_cmap('viridis', num_colors)
        colors_list = []
        for i in range(points.shape[0]):
            colors_list.append(color_map(points[i,3]))
        colors_list = np.array(colors_list)[:,0:3]
        pts.colors = open3d.utility.Vector3dVector(colors_list)
    elif point_colors is None:
        num_colors = 100
        color_map = cm.get_cmap('viridis', num_colors)
        dis_list = (points[:, 0]**2+points[:, 1]**2+points[:, 2]**2)**(1/2)
        dis_max = np.max(dis_list)
        colors_list = []
        for dis in dis_list:
            colors_list.append(color_map(dis/dis_max))
        colors_list = np.array(colors_list)[:,0:3]
        pts.colors = open3d.utility.Vector3dVector(colors_list)
        # c1 = [0.09411764705882353, 0.27058823529411763, 0.23137254901960785]
        # pts.colors = open3d.utility.Vector3dVector([c1 for p in pts.points])        
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        #vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)
        vis = draw_box(vis, ref_boxes, (0, 0, 1))

    vis.run()
    vis.destroy_window()


def draw_scenes2preds(points, gt_boxes=None, ref_boxes=None, ref_boxes2=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()
    if isinstance(ref_boxes2, torch.Tensor):
        ref_boxes2 = ref_boxes2.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 2.0    
    vis.get_render_option().background_color = np.ones(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        num_colors = 100
        color_map = cm.get_cmap('viridis', num_colors)
        dis_list = (points[:, 0]**2+points[:, 1]**2+points[:, 2]**2)**(1/2)
        dis_max = np.max(dis_list)
        colors_list = []
        for dis in dis_list:
            colors_list.append(color_map(dis/dis_max))
        colors_list = np.array(colors_list)[:,0:3]
        pts.colors = open3d.utility.Vector3dVector(colors_list)

        # c1 = [0.09411764705882353, 0.27058823529411763, 0.23137254901960785]
        # pts.colors = open3d.utility.Vector3dVector([c1 for p in pts.points])        
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        #vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)
        vis = draw_box(vis, ref_boxes, (0, 0, 1))

    if ref_boxes2 is not None:
        #vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)
        vis = draw_box(vis, ref_boxes2, (0, 1, 0))

    vis.run()
    vis.destroy_window()



def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
    

def draw_scenes2pts(points1, points2, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True):
    if isinstance(points1, torch.Tensor):
        points1 = points1.cpu().numpy()
    if isinstance(points2, torch.Tensor):
        points2 = points2.cpu().numpy()
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 5.0
    vis.get_render_option().background_color = np.ones(3)

    pts1 = open3d.geometry.PointCloud()
    pts1.points = open3d.utility.Vector3dVector(points1[:, :3])

    pts2 = open3d.geometry.PointCloud()
    pts2.points = open3d.utility.Vector3dVector(points2[:, :3])
    
    vis.add_geometry(pts1)
    color1 = np.zeros((points1.shape[0], 3))
    color1[:,0] = 1
    pts1.colors = open3d.utility.Vector3dVector(color1)
    
    vis.add_geometry(pts2)    
    color2 = np.zeros((points2.shape[0], 3))
    color2[:,2] = 1
    pts2.colors = open3d.utility.Vector3dVector(color2)

    #
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        #vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)
        vis = draw_box(vis, ref_boxes, (0, 0, 1))

    vis.run()
    vis.destroy_window()

def draw_points(points1, points2):
    if isinstance(points1, torch.Tensor):
        points1 = points1.cpu().numpy()
    if isinstance(points2, torch.Tensor):
        points2 = points2.cpu().numpy()
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 2.0
    vis.get_render_option().background_color = np.ones(3)

    pts1 = open3d.geometry.PointCloud()
    pts1.points = open3d.utility.Vector3dVector(points1[:, :3])

    pts2 = open3d.geometry.PointCloud()
    pts2.points = open3d.utility.Vector3dVector(points2[:, :3])
    
    vis.add_geometry(pts1)
    color1 = np.zeros((points1.shape[0], 3))
    #color1[:,0] = 1
    pts1.colors = open3d.utility.Vector3dVector(color1)
    
    vis.add_geometry(pts2)    
    color2 = np.zeros((points2.shape[0], 3))
    color2[:,1] = 1
    pts2.colors = open3d.utility.Vector3dVector(color2)

    vis.run()
    vis.destroy_window()


def draw_scenes_save(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True, path=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 2.0    
    vis.get_render_option().background_color = np.ones(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        num_colors = 100
        color_map = cm.get_cmap('viridis', num_colors)
        dis_list = (points[:, 0]**2+points[:, 1]**2+points[:, 2]**2)**(1/2)
        dis_max = np.max(dis_list)
        colors_list = []
        for dis in dis_list:
            colors_list.append(color_map(dis/dis_max))
        colors_list = np.array(colors_list)[:,0:3]
        pts.colors = open3d.utility.Vector3dVector(colors_list)

        # c1 = [0.09411764705882353, 0.27058823529411763, 0.23137254901960785]
        # pts.colors = open3d.utility.Vector3dVector([c1 for p in pts.points])        
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (1, 0, 0))

    if ref_boxes is not None:
        #vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)
        vis = draw_box(vis, ref_boxes, (0, 0, 1))

    #vis.run()
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(path)
    vis.destroy_window()


def draw_pts_4d(points, draw_origin=True, path=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()

    vis = open3d.visualization.Visualizer()
    vis.create_window()

    vis.get_render_option().point_size = 2.0    
    vis.get_render_option().background_color = np.ones(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    vis.add_geometry(pts)

    num_colors = 100
    color_map = cm.get_cmap('hot', num_colors)
    colors = points[:,3]/np.max(points[:,3])
    colors_list = []
    for c in colors:
        colors_list.append(color_map(c))
    colors_list = np.array(colors_list)[:,0:3]
    pts.colors = open3d.utility.Vector3dVector(colors_list)

    vis.run()
    vis.destroy_window()