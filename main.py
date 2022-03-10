#!/usr/bin/env python
#!coding=utf-8
import numpy as np
import pickle
import open3d
import time

f2 = open('/home/seivl/kitti_test_open3d/5411.pkl','rb')
data2 = pickle.load(f2)

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def my_compute_box_3d(center, size, heading_angle):

    h = size[2]
    w = size[0]
    l = size[1]
    heading_angle = -heading_angle - np.pi / 2 

    center[2] = center[2] + h / 2
    R = rotz(1*heading_angle)
    l = l/2
    w = w/2
    h = h/2
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)


def main():

    # axis = open3d.create_mesh_coordinate_frame(size=1,origin=[0,0,0])

    vis = open3d.Visualizer()
    vis.create_window(window_name="kitti")
    vis.get_render_option().point_size = 10

    # ctr = vis.get_view_control()
    # ctr.set_lookat([0, 0, 0.3])
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    pointcloud = open3d.PointCloud()
    pointcloud.points = open3d.utility.Vector3dVector(data2["point"]) 
    pointcloud.colors = open3d.utility.Vector3dVector([[255, 255, 255] for _ in range(len(data2["point"]))]) 

    for i in range(len(data2["gt"])):
        bbox = data2["gt"][i]
        corners_3d = my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])

        bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[1, 0, 0] for _ in range(len(bbox_lines))] #red

        bbox = open3d.LineSet()
        bbox.lines  = open3d.Vector2iVector(bbox_lines)
        bbox.colors = open3d.Vector3dVector(colors)
        bbox.points = open3d.Vector3dVector(corners_3d)
        vis.add_geometry(bbox)

    for i in range(len(data2["pred"])):
        bbox = data2["pred"][i]
        corners_3d = my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])

        bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[0, 1, 0] for _ in range(len(bbox_lines))]  #green

        bbox = open3d.LineSet()
        bbox.lines  = open3d.Vector2iVector(bbox_lines)
        bbox.colors = open3d.Vector3dVector(colors)
        bbox.points = open3d.Vector3dVector(corners_3d)
        vis.add_geometry(bbox)

    vis.add_geometry(pointcloud)
    # vis.add_geometry(axis)

    while True:
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
if __name__ == '__main__':
    main()

