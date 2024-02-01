import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

##############################################
############CAMERA ARCHITECTURE###############
##############################################

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=bias)
class Detection_Header(nn.Module):

    def __init__(self,config, use_bn=True, fourth_resnet_block=0):
        super(Detection_Header, self).__init__()

        self.use_bn = use_bn
        self.fourth_resnet_block = fourth_resnet_block
        self.config = config
        bias = not use_bn
        if config['model']['DetectionHead'] == 'True':
            self.conv1 = conv3x3(32, 16, bias=bias)
            self.bn1 = nn.BatchNorm2d(16)
            self.conv2 = conv3x3(16, 16, bias=bias)
            self.bn2 = nn.BatchNorm2d(16)
            self.conv3 = conv3x3(16, 16, bias=bias)
            self.bn3 = nn.BatchNorm2d(16)
            self.clshead = conv3x3(16, 1, bias=True)

    def forward(self, x):
        if self.config['model']['DetectionHead'] == 'True':
            x = self.conv1(x)
            if self.use_bn:
                x = self.bn1(x)
            x = self.conv2(x)
            if self.use_bn:
                x = self.bn2(x)
            x = self.conv3(x)
            if self.use_bn:
                x = self.bn3(x)

            cls = torch.sigmoid(self.clshead(x))

            return cls

##############################################
############CAMERA ARCHITECTURE###############
##############################################

class Bottleneck_camera(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None, expansion=4):
        super(Bottleneck_camera, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion * planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.relu(residual + out)
        return out


class warmupblock(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size=1, use_bn=True):
        super(warmupblock, self).__init__()
        self.use_bn = use_bn

        self.conv1 = nn.Conv2d(in_layer, out_layer, kernel_size,
                               stride=(1, 1), padding=1, bias=(not use_bn))

        self.bn1 = nn.BatchNorm2d(out_layer)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        if self.use_bn:
            x1 = self.bn1(x1)
        x = self.relu(x1)
        return x


class FPN_BackBone_camera(nn.Module):

    def __init__(self, num_block, channels, block_expansion, use_bn=True):
        super(FPN_BackBone_camera, self).__init__()
        self.block_expansion = block_expansion
        self.use_bn = use_bn
        self.warmup = warmupblock(3, 32, kernel_size=3, use_bn=True)
        self.in_planes = 32

        self.conv = nn.Conv2d(self.in_planes, self.in_planes, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=False)

        # Residuall blocks
        self.block1 = self._make_layer(Bottleneck_camera, planes=channels[0], num_blocks=num_block[0])
        self.block2 = self._make_layer(Bottleneck_camera, planes=channels[1], num_blocks=num_block[1])
        self.block3 = self._make_layer(Bottleneck_camera, planes=channels[2], num_blocks=num_block[2])
        self.block4 = self._make_layer(Bottleneck_camera, planes=channels[3], num_blocks=num_block[3])


    def forward(self, x, features_radar):
        x = self.warmup(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Backbone
        features = {}
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)

        features['x0'] = x
        features['x1'] = x1
        features['x2'] = x2
        features['x3'] = x3
        features['x4'] = x4

        # features['x4'] = torch.cat([x4, radar_feature_x4], dim=1)

        radar_feature_x2 = F.interpolate(features_radar['x2'], (135, 240))
        radar_feature_x3 = F.interpolate(features_radar['x3'], (68, 120))
        radar_feature_x4 = F.interpolate(features_radar['x4'], (34, 60))

        features['x4_fused'] = torch.cat((features['x4'], radar_feature_x4), axis=1)
        features['x3_fused'] = torch.cat((features['x3'], radar_feature_x3), axis=1)
        features['x2_fused'] = torch.cat((features['x2'], radar_feature_x2), axis=1)


        return features

    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * self.block_expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * self.block_expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * self.block_expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample, expansion=self.block_expansion))
        self.in_planes = planes * self.block_expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1, expansion=self.block_expansion))
            self.in_planes = planes * self.block_expansion
        return nn.Sequential(*layers)  # this *layers will unpack the list


class BasicBlock_UpScaling(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_UpScaling, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out


class UpScaling(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = BasicBlock_UpScaling(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1) #.clone()
        return self.conv(x)


class cam_decoder(nn.Module):
    def __init__(self, ):
        super(cam_decoder, self).__init__()
        self.L4 = nn.Conv2d(896, 512, kernel_size=1, stride=1, padding=0)
        self.L3 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.L2 = nn.Conv2d(324, 128, kernel_size=1, stride=1, padding=0)
        self.up1 = (UpScaling(512, 256))
        self.up2 = (UpScaling(256, 128))
        self.up3 = (UpScaling(128, 64))
        self.up4 = (UpScaling(64, 32))

    def forward(self, features):
        T0 = features['x0']
        T1 = features['x1']
        T2 = features['x2_fused']
        T3 = features['x3_fused']
        T4 = features['x4_fused']

        T4 = self.L4(T4)
        T3 = self.L3(T3)
        T2 = self.L2(T2)

        x = self.up1(T4, T3)
        x = self.up2(x, T2)
        x = self.up3(x, T1)
        out = self.up4(x, T0)
        return out

class cameraonly_perspective(nn.Module):
    def __init__(self, channels_bev, blocks):

        super(cameraonly_perspective, self).__init__()

        self.FPN = FPN_BackBone_camera(num_block=blocks, channels=channels_bev, block_expansion=4, use_bn=True)
        self.cam_decoder = cam_decoder()

    def forward(self, x, radar_feature_x4):
        features = self.FPN(x, radar_feature_x4)
        cam_decoded = self.cam_decoder(features)
        return cam_decoded


class fftradnet_adapted(nn.Module):
    def __init__(self, channels_bev, blocks, detection_head, segmentation_head,config):
        super(fftradnet_adapted, self).__init__()

        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        self.cameraencdec = cameraonly_perspective(channels_bev=channels_bev, blocks=blocks)

        if (self.detection_head):
            self.detection_header = Detection_Header(config=config,fourth_resnet_block=channels_bev[3] * 4)

        if (self.segmentation_head):
            self.freespace = nn.Sequential(BasicBlock_UpScaling(32, 32),
                                           nn.Conv2d(32, 1, kernel_size=1))


    def forward(self, cam_inputs, radar_feature_x4):

        out = {'Detection': [], 'Segmentation': []}
        decoder_output = self.cameraencdec(cam_inputs, radar_feature_x4)

        if (self.detection_head):
            out['Detection'] = self.detection_header(decoder_output)

        if (self.segmentation_head):
            frespace_pred = self.freespace(decoder_output)
            out['Segmentation'] = frespace_pred[:, :, :, :850]

        return out

#############################################################################################
######################################## radar ONLY #########################################
#############################################################################################

###############
# Basis Block #
###############

class BasisBlock(nn.Module):
    """
    BasisBlock for input to ResNet
    """

    def __init__(self, n_input_channels):
        super(BasisBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_input_channels, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x


#################
# Residual Unit #
#################

class ResidualUnit(nn.Module):
    def __init__(self, n_input, n_output, downsample=False):
        """
        Residual Unit consisting of two convolutional layers and an identity mapping
        :param n_input: number of input channels
        :param n_output: number of output channels
        :param downsample: downsample the output by a factor of 2
        """
        super(ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_output, n_output, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # down-sampling: use stride two for convolutional kernel and create 1x1 kernel for down-sampling of input
        self.downsample = None
        if downsample:
            self.conv1 = nn.Conv2d(n_input, n_output, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            self.downsample = nn.Sequential(nn.Conv2d(n_input, n_output, kernel_size=(1, 1), stride=(2, 2), bias=False),
                                            nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True,
                                                           track_running_stats=True))
        else:
            self.identity_channels = nn.Conv2d(n_input, n_output, kernel_size=(1, 1), bias=False)

    def forward(self, x):

        # store input for skip-connection
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        # downsample input to match output dimensions
        if self.downsample is not None:
            identity = self.downsample(identity)
        else:
            identity = self.identity_channels(identity)

        # skip-connection
        x += identity

        # apply ReLU activation
        x = self.relu(x)

        return x


##################
# Residual Block #
##################


class ResidualBlock(nn.Module):
    """
        Residual Block containing specified number of residual layers
        """

    def __init__(self, n_input, n_output, n_res_units):
        super(ResidualBlock, self).__init__()

        # use down-sampling only in the first residual layer of the block
        first_unit = True

        # specific channel numbers
        if n_res_units == 3:
            inputs = [n_input, n_output//4, n_output//4]
            outputs = [n_output//4, n_output//4, n_output]
        else:
            inputs = [n_input, n_output // 4, n_output // 4, n_output // 4, n_output // 4, n_output]
            outputs = [n_output // 4, n_output // 4, n_output // 4, n_output // 4, n_output, n_output]

        # create residual units
        units = []
        for unit_id in range(n_res_units):
            if first_unit:
                units.append(ResidualUnit(inputs[unit_id], outputs[unit_id], downsample=True))
                first_unit = False
            else:
                units.append(ResidualUnit(inputs[unit_id], outputs[unit_id]))
        self.res_block = nn.Sequential(*units)

    def forward(self, x):

        x = self.res_block(x)

        return x



#########
# PIXOR #
#########

class PIXOR(nn.Module):
    def __init__(self):
        super(PIXOR, self).__init__()

        # Backbone Network
        self.basis_block = BasisBlock(n_input_channels=46)
        self.res_block_1 = ResidualBlock(n_input=32, n_output=96, n_res_units=3)
        self.res_block_2 = ResidualBlock(n_input=96, n_output=196, n_res_units=6)
        self.res_block_3 = ResidualBlock(n_input=196, n_output=256, n_res_units=6)
        self.res_block_4 = ResidualBlock(n_input=256, n_output=384, n_res_units=3)


    def forward(self, x):
        features_radar = {}
        x_b = self.basis_block(x)
        #encoder for radar
        x_1 = self.res_block_1(x_b)
        x_2 = self.res_block_2(x_1)
        x_3 = self.res_block_3(x_2)
        x_4 = self.res_block_4(x_3)

        features_radar['x0'] = x
        features_radar['x1'] = x_1
        features_radar['x2'] = x_2
        features_radar['x3'] = x_3
        features_radar['x4'] = x_4
        
        return features_radar

################## FUSION x4 ####################
class cameraradar_fusion_x4(nn.Module):
    def __init__(self, channels_bev, blocks,detection_head,segmentation_head,config):
        super(cameraradar_fusion_x4, self).__init__()

        self.radaronly = PIXOR()
        self.cameraonly = fftradnet_adapted(channels_bev, blocks,detection_head,segmentation_head,config)

    def forward(self, cam_inputs, radar_inputs):

        features_radar = self.radaronly(radar_inputs)
        out = self.cameraonly(cam_inputs, features_radar)

        return out

# if __name__ == "__main__":
#     camera_input = torch.randn((2, 3, 540, 960))
#     radar_input = torch.randn((2, 46, 1030, 800))
#
#     net = cameraradar_fusion_x4(channels_bev=[16, 32, 64, 128],
#                                 blocks=[3, 6, 6, 3],
#                                 detection_head=True,segmentation_head=True)
#
#     fusion_afterdecoder = net(camera_input, radar_input)
