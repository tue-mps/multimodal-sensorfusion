import numpy as np
import cv2
import os
import pandas as pd
from .config import *

calib = np.load('/home/kavin/code_experiments/05_MultiTask_Fusion_v1/dataset/camera_calib.npy',allow_pickle=True).item()

def ConpensateLayerAngle(pcl, index, sensor_height):
    offset = 0
    if (index % 2 == 0):
        offset = np.deg2rad(.6)

    x = pcl[:, 4] * np.cos(pcl[:, 5] + offset) * np.cos(pcl[:, 6])
    y = pcl[:, 4] * np.cos(pcl[:, 5] + offset) * np.sin(pcl[:, 6])
    z = pcl[:, 4] * np.sin(pcl[:, 5] + offset) + sensor_height

    pcl[:, 0] = x
    pcl[:, 1] = y
    pcl[:, 2] = z

    return pcl

def read_im_laserpcl(image, pts,index):

    pts = ConpensateLayerAngle(pts,index,0.42)[:,:3]
    pts[:,[0, 1, 2]] = pts[:,[1, 0,2]] # Swap the order
    pts[:,0]*=-1 # Left is positive

    imgpts, _ = cv2.projectPoints(np.array(pts),
                                  calib['extrinsic']['rotation_vector'],
                                  calib['extrinsic']['translation_vector'],
                                  calib['intrinsic']['camera_matrix'],
                                calib['intrinsic']['distortion_coefficients'])

    imgpts=imgpts.squeeze(1).astype('int')

    # Keep only points inside the image size
    idx = np.where((imgpts[:, 0] >= 0) & (imgpts[:, 0] < image.shape[1]) &
                   (imgpts[:, 1] >= 0) & (imgpts[:, 1] < image.shape[0]))[0]

    return idx, imgpts


def read_im_laserpcl_with_GT(image, pts, lidar_gt, index):
    #[X, Y, Z, intensity, radialDistance, elevation_Angle, azimuth_angle, layer_index]
    pts = ConpensateLayerAngle(pts,index,0.42)[:,:3]
    pts[:,[0, 1, 2]] = pts[:,[1, 0,2]] # Swap the order
    pts[:,0]*=-1 # Left is positive
    # pts__ = np.round(pts,2)
    imgpts, _ = cv2.projectPoints(np.array(pts),
                                  calib['extrinsic']['rotation_vector'],
                                  calib['extrinsic']['translation_vector'],
                                  calib['intrinsic']['camera_matrix'],
                                  calib['intrinsic']['distortion_coefficients'])

    imgpts=imgpts.squeeze(1).astype('int')

    # Keep only points inside the image size
    idx = np.where((imgpts[:, 0] >= 0) & (imgpts[:, 0] < image.shape[1]) &
                   (imgpts[:, 1] >= 0) & (imgpts[:, 1] < image.shape[0]))[0]

    # here we project the lidar labels from labels.csv file to camera image
    imgpts_gt, _ = cv2.projectPoints(np.array(lidar_gt),
                                  calib['extrinsic']['rotation_vector'],
                                  calib['extrinsic']['translation_vector'],
                                  calib['intrinsic']['camera_matrix'],
                                  calib['intrinsic']['distortion_coefficients'])

    imgpts_gt = imgpts_gt.squeeze(1).astype('int')

    # Keep only points inside the image size
    idx_gt = np.where((imgpts_gt[:, 0] >= 0) & (imgpts_gt[:, 0] < image.shape[1]) &
                      (imgpts_gt[:, 1] >= 0) & (imgpts_gt[:, 1] < image.shape[0]))[0]

    return idx, imgpts, idx_gt, imgpts_gt


########################
# Voxelize Point Cloud #
########################


def voxelize(point_cloud):
    """
    Transform a continuous point cloud into a discrete voxelized grid that serves as the network input
    :param point_cloud: continuous point cloud | dim_0: all points, dim_1: [x, y, z, reflection]
    :return: voxelized point cloud | shape: [INPUT_DIM_0, INPUT_DIM_1, INPUT_DIM_2]
    """

    # remove all points outside the pre-specified FOV
    idx = np.where(point_cloud[:, 0] > VOX_X_MIN)
    point_cloud = point_cloud[idx]
    idx = np.where(point_cloud[:, 0] < VOX_X_MAX)
    point_cloud = point_cloud[idx]
    idx = np.where(point_cloud[:, 1] > VOX_Y_MIN)
    point_cloud = point_cloud[idx]
    idx = np.where(point_cloud[:, 1] < VOX_Y_MAX)
    point_cloud = point_cloud[idx]
    idx = np.where(point_cloud[:, 2] > VOX_Z_MIN)
    point_cloud = point_cloud[idx]
    idx = np.where(point_cloud[:, 2] < VOX_Z_MAX)
    point_cloud = point_cloud[idx]

    # create separate vectors for x, y, z coordinates and the reflectance value
    pxs = point_cloud[:, 0]
    pys = point_cloud[:, 1]
    pzs = point_cloud[:, 2]
    prs = point_cloud[:, 3]

    # convert velodyne coordinates to voxel
    qxs = (INPUT_DIM_0 - 1 - ((pxs - VOX_X_MIN) // VOX_X_DIVISION)).astype(np.int32)
    qys = ((-pys - VOX_Y_MIN) // VOX_Y_DIVISION).astype(np.int32)
    qzs = ((pzs - VOX_Z_MIN) // VOX_Z_DIVISION).astype(np.int32)
    quantized = np.dstack((qxs, qys, qzs, prs)).squeeze()

    # create empty voxel grid and reflectance image
    voxel_grid = np.zeros(shape=(INPUT_DIM_0, INPUT_DIM_1, INPUT_DIM_2-1), dtype=np.float32)
    reflectance_image = np.zeros(shape=(INPUT_DIM_0, INPUT_DIM_1), dtype=np.float32)
    reflectance_count = np.zeros(shape=(INPUT_DIM_0, INPUT_DIM_1), dtype=np.float32)

    # iterate over each point to fill occupancy grid and compute reflectance image
    for point_id, point in enumerate(quantized):
        point = point.astype(np.int32)
        voxel_grid[point[0], point[1], point[2]] = 1
        reflectance_image[point[0], point[1]] += point[3]
        reflectance_count[point[0], point[1]] += 1

    # take average over reflection of xy position
    reflectance_count = np.where(reflectance_count == 0, 1, reflectance_count)
    reflectance_image /= reflectance_count

    voxel_output = np.dstack((voxel_grid, reflectance_image))

    return voxel_output
