import json
import argparse
import torch
import numpy as np

######Standalone operation#######
from model.standalone.only_camera import cam_only

######Perspective Fusion#######
from model.fusion.perspective.earlyfusion import earlyfusion
from model.fusion.perspective.cameralidar_ad_fusion import cameralidar_fusion_Afterdecoder
from model.fusion.perspective.cameralidar_x4_fusion import cameralidar_fusion_x4
from model.fusion.perspective.cameraradar_ad_fusion import cameraradar_fusion_Afterdecoder
from model.fusion.perspective.cameraradar_x4_fusion import cameraradar_fusion_x4

######BEV Fusion#######
from model.fusion.bev.earlyfusion import earlyfusion_bev
from model.fusion.bev.cameraradar_ad_fusion import cameraradar_fusion_Afterdecoder_bev
from model.fusion.bev.cameraradar_x4_fusion import cameraradar_fusion_x4_bev

from dataset.encoder import ra_encoder
from dataset.dataset_fusion import RADIal
from dataset.dataloader_fusion import CreateDataLoaders
import cv2
from utils.util import DisplayHMI
import time

gpu_id = 0

def main(config, checkpoint_filename):

    # set device
    device = torch.device('cuda:' + str(gpu_id) if torch.cuda.is_available() else 'cpu')
    print("Device used:", device)

    # load dataset and create model
    if config['model']['view_perspective'] == 'True':
        dataset = RADIal(config=config,
                         encoder=None,
                         difficult=True)
        train_loader, val_loader, test_loader = CreateDataLoaders(dataset, config, config['seed'])

        if config['architecture']['perspective']['only_camera'] == 'True':
            net = cam_only(channels_bev=config['model']['channels_bev'],
                           blocks=config['model']['backbone_block'],
                           detection_head=config['model']['DetectionHead'],
                           segmentation_head=config['model']['SegmentationHead'])
            print("***************************************")
            print("CameraOnly in front view has been chosen")
            print("***************************************")

        if config['architecture']['perspective']['early_fusion'] == 'True':
            net = earlyfusion(channels_bev=config['model']['channels_bev'],
                              blocks=config['model']['backbone_block'],
                              detection_head=config['model']['DetectionHead'],
                              segmentation_head = config['model']['SegmentationHead'],
                              config=config)
            print("***************************************")
            print("Early fusion in front view has been chosen")
            print("***************************************")
        if config['architecture']['perspective']['after_decoder_fusion'] == 'True':
            if (config['model']['camera_input'] == 'True' and config['model']['radar_input'] == 'True' and config['model']['lidar_input'] == 'False'):
                net = cameraradar_fusion_Afterdecoder(channels_bev=config['model']['channels_bev'],
                                                      blocks=config['model']['backbone_block'],
                                                      detection_head=config['model']['DetectionHead'],
                                                      segmentation_head=config['model']['SegmentationHead'])
                print("***************************************")
                print("CameraRadar AD fusion in front view has been chosen")
                print("***************************************")

            if (config['model']['camera_input'] == 'True' and config['model']['radar_input'] == 'False' and config['model']['lidar_input'] == 'True'):
                net = cameralidar_fusion_Afterdecoder(channels_bev=config['model']['channels_bev'],
                                                      blocks=config['model']['backbone_block'],
                                                      detection_head=config['model']['DetectionHead'],
                                                      segmentation_head = config['model']['SegmentationHead'],)
                print("***************************************")
                print("CameraLidar AD fusion in front view has been chosen")
                print("***************************************")

        if config['architecture']['perspective']['x4_fusion'] == 'True':
            if (config['model']['camera_input'] == 'True' and config['model']['radar_input'] == 'True' and config['model']['lidar_input'] == 'False'):
                net = cameraradar_fusion_x4(channels_bev=config['model']['channels_bev'],
                                            blocks=config['model']['backbone_block'],
                                            detection_head=config['model']['DetectionHead'],
                                            segmentation_head=config['model']['SegmentationHead'],
                                            )
                print("***************************************")
                print("CameraRadar x4 fusion in front view has been chosen")
                print("***************************************")

            if (config['model']['camera_input'] == 'True' and config['model']['radar_input'] == 'False' and config['model']['lidar_input'] == 'True'):
                net = cameralidar_fusion_x4(channels_bev=config['model']['channels_bev'],
                                            blocks=config['model']['backbone_block'],
                                            detection_head=config['model']['DetectionHead'],
                                            segmentation_head=config['model']['SegmentationHead'],
                                            )
                print("***************************************")
                print("CameraLidar x4 fusion in front view has been chosen")
                print("***************************************")

    if config['model']['view_birdseye'] == 'True':
        enc = ra_encoder(geometry=config['dataset']['geometry'],
                         statistics=config['dataset']['statistics'],
                         regression_layer=2)

        dataset = RADIal(config=config,
                         encoder=enc.encode,
                         difficult=True)

        train_loader, val_loader, test_loader = CreateDataLoaders(dataset, config, config['seed'])

        if config['architecture']['bev']['early_fusion'] == 'True':
            net = earlyfusion_bev(mimo_layer=config['model']['MIMO_output'],
                                channels=config['model']['channels'],
                                blocks=config['model']['backbone_block'],
                                detection_head=config['model']['DetectionHead'],
                                radar_input=config['model']['radar_input'],
                                camera_input=config['model']['camera_input'],
                                fusion=config['architecture']['early_fusion'])
            print("***************************************")
            print("CameraRadar early fusion in BEV has been chosen")
            print("***************************************")

        if config['architecture']['bev']['x4_fusion'] == 'True':
            net = cameraradar_fusion_x4_bev(mimo_layer=config['model']['MIMO_output'],
                                   channels=config['model']['channels'],
                                   channels_bev=config['model']['channels_bev'],
                                   blocks=config['model']['backbone_block'],
                                   detection_head=config['model']['DetectionHead'],
                                   radar_input=config['model']['radar_input'],
                                   camera_input=config['model']['camera_input'],
                                   fusion=config['architecture']['x4_fusion'])

            print("***************************************")
            print("CameraRadar x4 fusion in BEV has been chosen")
            print("***************************************")

        if config['architecture']['bev']['after_decoder_fusion'] == 'True':
            net = cameraradar_fusion_Afterdecoder_bev(mimo_layer=config['model']['MIMO_output'],
                                       channels=config['model']['channels'],
                                       channels_bev=config['model']['channels_bev'],
                                       blocks=config['model']['backbone_block'],
                                       detection_head=config['model']['DetectionHead'],
                                       radar_input=config['model']['radar_input'],
                                       camera_input=config['model']['camera_input'],
                                       fusion=config['architecture']['after_decoder_fusion'])

            print("*************************************************")
            print("CameraRadar AD fusion in BEV has been chosen")
            print("*************************************************")

    net.to(device)

    # Load the model
    dict = torch.load(checkpoint_filename, map_location=device)
    net.load_state_dict(dict['net_state_dict'])
    net.eval()

    for data in test_loader:

        if config['architecture']['perspective']['only_camera'] == 'True' or config['architecture']['perspective']['early_fusion'] == 'True':
            # fusion is done on dataset.py
            inputs1 = data[0].to(device).float()
            seg_map_label = data[1].to(device).double()
            det_label = data[2].to(device).float()
            with torch.set_grad_enabled(False):
                outputs = net(inputs1)

        if (config['architecture']['perspective']['after_decoder_fusion'] == 'True' or
                config['architecture']['perspective']['x4_fusion'] == 'True' or config['model']['view_birdseye'] == 'True'):
            inputs1 = data[0].to(device).float()
            inputs2 = data[1].to(device).float()
            seg_map_label = data[2].to(device).double()
            det_label = data[3].to(device).float()
            with torch.set_grad_enabled(False):
                outputs = net(inputs2, inputs1)

        hmi = DisplayHMI(data[3], seg_map_label, det_label,outputs)
        # resized_hmi = cv2.resize(hmi, (1600, 450))

        cv2.imshow('Multi-Tasking',hmi)
        # cv2.imwrite('/media/kach271771/LocalDisk/PhD/dataset/Valeo/sample_objdetection/cameraonly/' + data[5][-16:], hmi*255)
        # cv2.waitKey(0)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Evaluation')
    parser.add_argument('-c', '--config',
                        default='/media/kach271771/LocalDisk/PhD/VM_data/RADIal_experiments/05_MultiTask_Fusion/results_pth/perspective/cameraradarlidar/config_allmodality.json',
                        type=str,
                        help='Path to the config file (default: config_allmodality.json)')
    parser.add_argument('-r', '--checkpoint',
                        default="/media/kach271771/LocalDisk/PhD/VM_data/RADIal_experiments/05_MultiTask_Fusion/results_pth/perspective/cameraradarlidar/MultiTaskFusion_CameraRadarLidarER_epoch96_loss_2607277.1652_AP_0.9968_AR_0.9512_IOU_0.9636.pth",
                        type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint)