import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

calib = np.load('/home/kavin/code_experiments/05_MultiTask_Fusion_v1/dataset/camera_calib.npy',allow_pickle=True).item()

def rotation2d(xyz,roll,yaw,pitch):

    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)

    rotation_vector = np.array([roll,pitch,yaw])
    rotation = R.from_rotvec(rotation_vector)
    rotated_vec = rotation.apply(xyz)

    return rotated_vec[:,:3]

def read_im_radarpcl(image, pts):
    Range = pts[:,0]
    Azimuth = pts[:, 1]
    Elevation = pts[:, 2]
    pts = np.stack([Range, Azimuth, Elevation], axis=1)
    pts = rotation2d(pts, 0, 0, -2)
    x = -pts[:, 1]  # lateral
    y = pts[:, 0]  # longi
    z = pts[:, 2]  # longi
    objectPoints = np.stack([x, y, z], axis=1)

    imgpts, _ = cv2.projectPoints(objectPoints,
                                  calib['extrinsic']['rotation_vector'],
                                  calib['extrinsic']['translation_vector'],
                                  calib['intrinsic']['camera_matrix'],
                                calib['intrinsic']['distortion_coefficients'])

    imgpts=imgpts.squeeze(1).astype('int')

    # Keep only points inside the image size
    idx = np.where((imgpts[:, 0] >= 0) & (imgpts[:, 0] < image.shape[1]) &
                   (imgpts[:, 1] >= 0) & (imgpts[:, 1] < image.shape[0]))[0]
    return idx, imgpts