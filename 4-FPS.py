import json
import argparse
import torch
######Standalone operation#######
from model.standalone.only_camera import cam_only
from model.standalone.FFTRadNet import FFTRadNet #radar only using RD spectrum

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
import time
from dataset.dataloader_fusion import CreateDataLoaders


def calculate_fps(model, inputs1):
    start_time = time.time()
    for i in range(100):
        model(inputs1)
    end_time = time.time()
    fps = 100 / (end_time - start_time)
    return fps

def calculate_fps_fusion(model, inputs1, inputs2):
    start_time = time.time()
    for i in range(100):
        model(inputs2, inputs1)
    end_time = time.time()
    fps = 100 / (end_time - start_time)
    return fps

def main(config, checkpoint_filename):
    # set device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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
                           segmentation_head=config['model']['SegmentationHead'],
                           config=config)
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
                                                      segmentation_head=config['model']['SegmentationHead'],
                                                      config=config,)
                print("***************************************")
                print("CameraRadar AD fusion in front view has been chosen")
                print("***************************************")

            if (config['model']['camera_input'] == 'True' and config['model']['radar_input'] == 'False' and config['model']['lidar_input'] == 'True'):
                net = cameralidar_fusion_Afterdecoder(channels_bev=config['model']['channels_bev'],
                                                      blocks=config['model']['backbone_block'],
                                                      detection_head=config['model']['DetectionHead'],
                                                      segmentation_head = config['model']['SegmentationHead'],
                                                      config=config,)
                print("***************************************")
                print("CameraLidar AD fusion in front view has been chosen")
                print("***************************************")

        if config['architecture']['perspective']['x4_fusion'] == 'True':
            if (config['model']['camera_input'] == 'True' and config['model']['radar_input'] == 'True' and config['model']['lidar_input'] == 'False'):
                net = cameraradar_fusion_x4(channels_bev=config['model']['channels_bev'],
                                            blocks=config['model']['backbone_block'],
                                            detection_head=config['model']['DetectionHead'],
                                            segmentation_head=config['model']['SegmentationHead'],
                                            config=config,
                                            )
                print("***************************************")
                print("CameraRadar x4 fusion in front view has been chosen")
                print("***************************************")

            if (config['model']['camera_input'] == 'True' and config['model']['radar_input'] == 'False' and config['model']['lidar_input'] == 'True'):
                net = cameralidar_fusion_x4(channels_bev=config['model']['channels_bev'],
                                            blocks=config['model']['backbone_block'],
                                            detection_head=config['model']['DetectionHead'],
                                            segmentation_head=config['model']['SegmentationHead'],
                                            config=config,
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

        if config['architecture']['bev']['only_radar'] == 'True':
            net = FFTRadNet(blocks=config['model']['backbone_block'],
                            mimo_layer=config['model']['MIMO_output'],
                            channels=config['model']['channels'],
                            detection_head=config['model']['DetectionHead'],
                            segmentation_head=config['model']['SegmentationHead'],
                            config=config,
                            regression_layer=2)
            print("***************************************")
            print("Only Radar (FFTRadNet) has been chosen")
            print("***************************************")

        if config['architecture']['bev']['early_fusion'] == 'True':
            net = earlyfusion_bev(mimo_layer=config['model']['MIMO_output'],
                                channels=config['model']['channels'],
                                blocks=config['model']['backbone_block'],
                                detection_head=config['model']['DetectionHead'],
                                segmentation_head=config['model']['SegmentationHead'],
                                config=config,
                                regression_layer=2)
            print("***************************************")
            print("CameraRadar early fusion in BEV has been chosen")
            print("***************************************")

        if config['architecture']['bev']['x4_fusion'] == 'True':
            net = cameraradar_fusion_x4_bev(mimo_layer=config['model']['MIMO_output'],
                                           channels=config['model']['channels'],
                                           channels_bev=config['model']['channels_bev'],
                                           blocks=config['model']['backbone_block'],
                                           detection_head=config['model']['DetectionHead'],
                                           segmentation_head=config['model']['SegmentationHead'],
                                           config=config,
                                           regression_layer=2)

            print("***************************************")
            print("CameraRadar x4 fusion in BEV has been chosen")
            print("***************************************")

        if config['architecture']['bev']['after_decoder_fusion'] == 'True':
            net = cameraradar_fusion_Afterdecoder_bev(mimo_layer=config['model']['MIMO_output'],
                                                      channels=config['model']['channels'],
                                                      channels_bev=config['model']['channels_bev'],
                                                      blocks=config['model']['backbone_block'],
                                                      detection_head=config['model']['DetectionHead'],
                                                      segmentation_head=config['model']['SegmentationHead'],
                                                      config=config,
                                                      regression_layer=2)

            print("*************************************************")
            print("CameraRadar AD fusion in BEV has been chosen")
            print("*************************************************")

    net.to(device)

    # Load the model
    dict = torch.load(checkpoint_filename, map_location=device)
    net.load_state_dict(dict['net_state_dict'])
    net.eval()

    for idx, data in enumerate(test_loader):

        if (config['architecture']['perspective']['only_camera'] == 'True' or
                config['architecture']['perspective']['early_fusion'] == 'True' or
                config['architecture']['bev']['only_radar'] == 'True'):
            inputs1 = data[0].to(device).float()
            fps = calculate_fps(net, inputs1)

        if (config['architecture']['perspective']['after_decoder_fusion'] == 'True' or
                config['architecture']['perspective']['x4_fusion'] == 'True' or
                config['architecture']['bev']['early_fusion'] == 'True'or
                config['architecture']['bev']['after_decoder_fusion'] == 'True' or
                config['architecture']['bev']['x4_fusion'] == 'True'):
            inputs1 = data[0].to(device).float()
            inputs2 = data[1].to(device).float()
            fps = calculate_fps_fusion(net, inputs1, inputs2)

        print(f"FPS: {fps:.2f}")

        if idx == 5:  # 6 iterations (since Python is zero-indexed)
            break


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('-c', '--config',
                        default="/home/kavin/code_experiments/05_resultspth_allmodality/results_pth/bev/onlyradar/OnlySegmentation_OnlyRadar_RD_fftradnet___Nov-30-2023___09:21:09/config_allmodality.json",
                        help='Path to the config file (default: config_allmodality.json)')
    parser.add_argument('-r', '--checkpoint',
                        default="/home/kavin/code_experiments/05_resultspth_allmodality/results_pth/bev/onlyradar/OnlySegmentation_OnlyRadar_RD_fftradnet___Nov-30-2023___09:21:09/OnlySegmentation_OnlyRadar_RD_fftradnet_epoch98_loss_111672.2928_IOU_0.6620.pth",
                        type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint)
