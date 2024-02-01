from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from torchvision.transforms import Resize,CenterCrop
import torchvision.transforms as transform
import pandas as pd
from PIL import Image
from .lidar_func import ConpensateLayerAngle, read_im_laserpcl, voxelize
from .radar_func import read_im_radarpcl

class RADIal(Dataset):

    def __init__(self, config, encoder=None,difficult=True):

        self.config = config
        self.encoder = encoder
        root_dir = self.config['dataset']['root_dir']
        self.labels = pd.read_csv(os.path.join(root_dir,'labels.csv')).to_numpy()
       
        # Keeps only easy samples
        # (this means all 1's (difficult samples) in the
        # last column of the csv file are ignored!
        # When difficult=True, then whole dataset is being considered (8252 samples)
        if(difficult==False):
            ids_filters=[]
            ids = np.where(self.labels[:, -1] == 0)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.labels = self.labels[ids_filters]

        # Gather each input entries by their sample id
        self.unique_ids = np.unique(self.labels[:,0])
        self.label_dict = {}
        for i,ids in enumerate(self.unique_ids):
            sample_ids = np.where(self.labels[:,0]==ids)[0]
            self.label_dict[ids]=sample_ids
        self.sample_keys = list(self.label_dict.keys())

        self.resize = Resize((256,224), interpolation=transform.InterpolationMode.NEAREST)
        self.crop = CenterCrop((512,448))


    def __len__(self):
        return len(self.label_dict)

    def __getitem__(self, index):

        root_dir = self.config['dataset']['root_dir']
        processed_dir = self.config['dataset']['processed_data']
        statistics = self.config['dataset']['statistics']

        perspective_view = self.config['model']['view_perspective']
        birdseye_view = self.config['model']['view_birdseye']

        camerainput = self.config['model']['camera_input']
        radarinput = self.config['model']['radar_input']
        lidarinput = self.config['model']['lidar_input']

        #PerspectiveView
        onlycamera_p = self.config['architecture']['perspective']['only_camera']
        earlyfusion_p = self.config['architecture']['perspective']['early_fusion']
        x4fusion_p = self.config['architecture']['perspective']['x4_fusion']
        adfusion_p = self.config['architecture']['perspective']['after_decoder_fusion']

        # bev
        onlyradar_b = self.config['architecture']['bev']['only_radar']
        earlyfusion_b = self.config['architecture']['bev']['early_fusion']
        x4fusion_b = self.config['architecture']['bev']['x4_fusion']
        adfusion_b = self.config['architecture']['bev']['after_decoder_fusion']

        # Get the sample id
        sample_id = self.sample_keys[index] 

        # From the sample id, retrieve all the labels ids
        entries_indexes = self.label_dict[sample_id]

        # Get the objects labels
        box_labels = self.labels[entries_indexes]

        # Labels contains following parameters:
        # x1_pix	y1_pix	x2_pix	y2_pix	laser_X_m	laser_Y_m	laser_Z_m radar_X_m	radar_Y_m	radar_R_m

        # format as following [Range, Angle, Doppler,laser_X_m,laser_Y_m,laser_Z_m,x1_pix,y1_pix,x2_pix	,y2_pix, radar_X_m, radar_Y_m]
        box_labels = box_labels[:,[10,11,12,5,6,7,1,2,3,4,8,9]].astype(np.float32)

        # when I use front view cam image and front view labels, then it goes here...
        if perspective_view == 'True':
            # Detection labels
            if (self.encoder == None):
                channels, height, width = 1, 540, 960
                out_label = np.zeros((channels, height, width), dtype=np.uint8)
                for lab in box_labels:
                    if (lab[6] == -1):
                        continue
                    x1, y1, x2, y2 = lab[6], lab[7], lab[8], lab[9]
                    # Convert to integer and ensure coordinates are within image dimensions
                    x1, y1, x2, y2 = int(x1/2), int(y1/2), int(x2/2), int(y2/2)
                    out_label[:, y1:y2, x1:x2] = 1
            else:
                out_label = 0

            # Segmentation labels
            segmap_name = os.path.join(processed_dir, 'camera_Freespace', "freespace_{:06d}.png".format(sample_id))
            segmap = Image.open(segmap_name)
            segmap = np.asarray(segmap) == 255

            if onlycamera_p == 'True':
                cam_img_name = os.path.join(root_dir, 'camera', "image_{:06d}.jpg".format(sample_id))
                # img_left_area = (0, 0, 850, 540)
                # crop_camim = Image.open(cam_img_name)
                # cameraimage = np.asarray(crop_camim.crop(img_left_area))
                cameraimage = cv2.imread(cam_img_name)
                return cameraimage, segmap, out_label, box_labels, cam_img_name

            if earlyfusion_p == 'True':
                if (radarinput == 'True' and camerainput == 'True' and lidarinput == 'False'):
                    # Read the camera image for radar camera fusion
                    img_name_ = os.path.join(root_dir, 'camera', "image_{:06d}.jpg".format(sample_id))
                    img_name = cv2.imread(img_name_)
                    new_size = (1920, 1080)
                    resized_image = cv2.resize(img_name, new_size)
                    # radar_cam = resized_image.copy()

                    # Read radar point clouds
                    radar_name = os.path.join(root_dir, 'radar_PCL', "pcl_{:06d}.npy".format(sample_id))
                    rad_point_cloud_data = np.load(radar_name)
                    rad_point_cloud_data = np.transpose(rad_point_cloud_data)

                    idx, imgpts = read_im_radarpcl(image=resized_image, pts=rad_point_cloud_data)
                    radar_point_cloud = np.zeros((1080,1920), dtype=np.float32)
                    for pt in imgpts:
                        if (pt[0] > 0 and pt[0] < 1920 and pt[1] > 0 and pt[1] < 1080):
                            center = [int(pt[0]),int(pt[1])]
                            radar_point_cloud[center[1], center[0]] = 255
                            # cv2.circle(radar_cam, center, 3, (0,0,255), -1)
                    radar_cam = np.dstack((resized_image, radar_point_cloud))
                    new_size = (960, 540)
                    radar_cam = cv2.resize(radar_cam, new_size)

                    return radar_cam, segmap, out_label, img_name_

                if (lidarinput == 'True' and camerainput == 'True' and
                    radarinput == 'False'):
                    # Read the camera image
                    img_name_ = os.path.join(root_dir, 'camera', "image_{:06d}.jpg".format(sample_id))
                    img_name = cv2.imread(img_name_)
                    new_size = (1920, 1080)
                    resized_image = cv2.resize(img_name, new_size)
                    # lidar_cam = resized_image.copy()

                    # Read the LiDAR point clouds
                    lidar_name = os.path.join(root_dir, 'laser_PCL', "pcl_{:06d}.npy".format(sample_id))
                    li_point_cloud_data = np.load(lidar_name)

                    idx, imgpts = read_im_laserpcl(image=resized_image,
                                                   pts=li_point_cloud_data,
                                                   index=index)
                    lidar_point_cloud = np.zeros((1080, 1920), dtype=np.float32)
                    for pt in imgpts:
                        if (pt[0] > 0 and pt[0] < 1920 and pt[1] > 0 and pt[1] < 1080):
                            center = [int(pt[0]), int(pt[1])]
                            # center = [int(imgpts[idx, 0]), int(imgpts[idx, 1])]
                            lidar_point_cloud[center[1], center[0]] = 255
                            # lidar_cam = cv2.circle(lidar_cam,(int(imgpts[idx, 0]),int(imgpts[idx, 1])), 3, (0, 0, 255), -1)
                    lidar_cam = np.dstack((resized_image, lidar_point_cloud))
                    new_size = (960, 540)
                    lidar_cam = cv2.resize(lidar_cam, new_size)

                    return lidar_cam, segmap, out_label, img_name_

                if (radarinput == 'True' and lidarinput == 'True' and
                    camerainput == 'True'):
                    # Read the camera image
                    img_name_ = os.path.join(root_dir, 'camera', "image_{:06d}.jpg".format(sample_id))
                    img_name = cv2.imread(img_name_)
                    new_size = (1920, 1080)
                    resized_image = cv2.resize(img_name, new_size)
                    # camera_radar_lidar = resized_image.copy()

                    # Read the LiDAR point clouds
                    lidar_name = os.path.join(root_dir, 'laser_PCL', "pcl_{:06d}.npy".format(sample_id))
                    li_point_cloud_data = np.load(lidar_name)

                    idx_li, imgpts_li = read_im_laserpcl(image=resized_image,
                                                   pts=li_point_cloud_data,
                                                   index=index)

                    lidar_point_cloud = np.zeros((1080, 1920), dtype=np.float32)
                    for pt in imgpts_li:
                        if (pt[0] > 0 and pt[0] < 1920 and pt[1] > 0 and pt[1] < 1080):
                            center = [int(pt[0]), int(pt[1])]
                            lidar_point_cloud[center[1], center[0]] = 255

                    # Read the Radar point clouds
                    radar_name = os.path.join(root_dir, 'radar_PCL', "pcl_{:06d}.npy".format(sample_id))
                    radar_point_cloud_data = np.load(radar_name)
                    radar_point_cloud_data = np.transpose(radar_point_cloud_data)
                    idx_rad, imgpts_rad = read_im_radarpcl(image=resized_image, pts=radar_point_cloud_data)

                    radar_point_cloud = np.zeros((1080, 1920), dtype=np.float32)
                    for pt in imgpts_rad:
                        if (pt[0] > 0 and pt[0] < 1920 and pt[1] > 0 and pt[1] < 1080):
                            center = [int(pt[0]), int(pt[1])]
                            radar_point_cloud[center[1], center[0]] = 255

                    camera_radar_lidar = np.dstack((resized_image, radar_point_cloud, lidar_point_cloud))
                    new_size = (960, 540)
                    camera_radar_lidar = cv2.resize(camera_radar_lidar, new_size)

                    return camera_radar_lidar, segmap, out_label, img_name_

            # here we use front view cam image and voxelized (BEV) radar/lidar data
            if adfusion_p == 'True' or x4fusion_p == 'True':
                if (radarinput == 'True' and camerainput == 'True' and lidarinput == 'False'):
                    img_name_ = os.path.join(root_dir, 'camera', "image_{:06d}.jpg".format(sample_id))
                    camera_image = cv2.imread(img_name_)
                    radar_name = os.path.join(root_dir, 'radar_PCL', "pcl_{:06d}.npy".format(sample_id))
                    rad_point_cloud_data = np.load(radar_name)
                    rad_point_cloud_data = np.transpose(rad_point_cloud_data)
                    x = rad_point_cloud_data[:, 5]
                    y = -rad_point_cloud_data[:, 6]
                    z = rad_point_cloud_data[:, 7]
                    intensity = rad_point_cloud_data[:, 4]
                    rad_pointcloud_xyz = np.stack([x, y, z, intensity], axis=1)
                    voxel_point_cloud = voxelize(rad_pointcloud_xyz)

                    return voxel_point_cloud, camera_image, segmap, out_label, img_name_

                #if you wanna train with both lidar pc and camera
                if (lidarinput == 'True' and camerainput == 'True' and
                    radarinput == 'False'):
                    # Read the camera image
                    img_name_ = os.path.join(root_dir, 'camera', "image_{:06d}.jpg".format(sample_id))
                    camera_image = cv2.imread(img_name_)
                    lidar_name = os.path.join(root_dir, 'laser_PCL', "pcl_{:06d}.npy".format(sample_id))
                    li_point_cloud_data = np.load(lidar_name)
                    li_pointcloud_xyz = ConpensateLayerAngle(li_point_cloud_data, index, 0.42)[:, :4]
                    voxel_point_cloud = voxelize(li_pointcloud_xyz)

                    return voxel_point_cloud, camera_image, segmap, out_label, img_name_

        # when I use BEV view cam image and BEV view labels, then it goes here...
        if birdseye_view == 'True':
            # Detection labels
            if (self.encoder != None):
                out_label = self.encoder(box_labels).copy()
            else:
                out_label = 0

            # Read the segmentation map in BEV
            segmap_name_polar = os.path.join(root_dir, 'radar_Freespace', "freespace_{:06d}.png".format(sample_id))
            segmap_polar = Image.open(segmap_name_polar)  # [512,900]
            # 512 pix for the range and 900 pix for the horizontal FOV (180deg)
            # We crop the fov to 89.6deg
            segmap_polar = self.crop(segmap_polar)
            # and we resize to half of its size
            segmap_polar = np.asarray(self.resize(segmap_polar)) == 255

            # Read the Radar FFT data
            radar_name = os.path.join(root_dir, 'radar_FFT', "fft_{:06d}.npy".format(sample_id))
            rd_input = np.load(radar_name, allow_pickle=True)
            radar_FFT = np.concatenate([rd_input.real, rd_input.imag], axis=2)
            if (statistics is not None):
                for i in range(len(statistics['input_mean'])):
                    radar_FFT[..., i] -= statistics['input_mean'][i]
                    radar_FFT[..., i] /= statistics['input_std'][i]

            bev_img_name = os.path.join(processed_dir, 'BEV_Polar_Python_resized', "image_{:06d}.jpg".format(sample_id))
            bev_image = np.asarray(Image.open(bev_img_name))
            if onlyradar_b == 'False':
                return radar_FFT, bev_image, segmap_polar, out_label, box_labels, bev_img_name
            else:
                return radar_FFT, segmap_polar, out_label, box_labels, bev_img_name