import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transform
import copy
import cv2

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


    def forward(self, x):
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
        self.up1 = (UpScaling(512, 256))
        self.up2 = (UpScaling(256, 128))
        self.up3 = (UpScaling(128, 64))
        self.up4 = (UpScaling(64, 32))

    def forward(self, features):
        T0 = features['x0']
        T1 = features['x1']
        T2 = features['x2']
        T3 = features['x3']
        T4 = features['x4']

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

    def forward(self, x):
        features = self.FPN(x)
        cam_decoded = self.cam_decoder(features)
        return cam_decoded

class fftradnet_adapted(nn.Module):
    def __init__(self, channels_bev, blocks):
        super(fftradnet_adapted, self).__init__()

        self.cameraencdec = cameraonly_perspective(channels_bev=channels_bev, blocks=blocks)

    def forward(self, cam_inputs):

        encdec_output = self.cameraencdec(cam_inputs)
        # encdec_output = F.interpolate(encdec_output, (258, 200))

        return encdec_output


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


#############
# FPN Block #
#############


class FPNBlock(nn.Module):
    """
        Block for Feature Pyramid Network including up-sampling and concatenation of feature maps
        """

    def __init__(self, bottom_up_channels, top_down_channels, fused_channels):
        super(FPNBlock, self).__init__()
        # reduce number of top-down channels to 196
        intermediate_channels = 196
        if top_down_channels > 196:
            self.channel_conv_td = nn.Conv2d(top_down_channels, intermediate_channels, kernel_size=(1, 1),
                                             stride=(1, 1), bias=False)
        else:
            self.channel_conv_td = None

        # change number of bottom-up channels to 128
        self.channel_conv_bu = nn.Conv2d(bottom_up_channels, fused_channels, kernel_size=(1, 1),
                                         stride=(1, 1), bias=False) #padding=(1,0)

        # transposed convolution on top-down feature maps
        if fused_channels == 128:
            out_pad = (0, 1)
        else:
            out_pad = (1, 1)
        if self.channel_conv_td is not None:
            self.deconv = nn.ConvTranspose2d(intermediate_channels, fused_channels, kernel_size=(3, 3), padding=(1, 1),
                                             stride=2, output_padding=out_pad)
        else:
            self.deconv = nn.ConvTranspose2d(top_down_channels, fused_channels, kernel_size=(3, 3), padding=(1, 1),
                                             stride=2, output_padding=out_pad)

    def forward(self, x_td, x_bu):

        # apply 1x1 convolutional to obtain required number of channels if needed
        if self.channel_conv_td is not None:
            x_td = self.channel_conv_td(x_td)

        # up-sample top-down feature maps
        x_td = self.deconv(x_td)

        # apply 1x1 convolutional to obtain required number of channels
        x_bu = self.channel_conv_bu(x_bu)

        diffY = x_bu.size()[2] - x_td.size()[2]
        diffX = x_bu.size()[3] - x_td.size()[3]

        x_td = F.pad(x_td, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        # perform element-wise addition
        x = x_td.add(x_bu)

        return x


####################
# Detection Header #
####################

class DetectionHeader(nn.Module):

    def __init__(self,config, n_input, n_output):
        super(DetectionHeader, self).__init__()
        self.config = config
        if config['model']['DetectionHead'] == 'True':
            basic_block = nn.Sequential(nn.Conv2d(n_input, n_output, kernel_size=(3, 3), padding=(1, 1), bias=False),
                                      nn.BatchNorm2d(n_output, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                      nn.ReLU(inplace=True))
            self.conv1 = basic_block
            self.conv2 = copy.deepcopy(basic_block)
            self.conv3 = copy.deepcopy(basic_block)
            self.conv4 = copy.deepcopy(basic_block)
            self.classification = nn.Conv2d(n_output, 1, kernel_size=(3, 3), padding=(1, 1))
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.config['model']['DetectionHead'] == 'True':
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            class_output = self.sigmoid(self.classification(x))

            return class_output

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

        # FPN blocks
        self.fpn_block_1 = FPNBlock(top_down_channels=384, bottom_up_channels=256, fused_channels=128)
        self.fpn_block_2 = FPNBlock(top_down_channels=128, bottom_up_channels=196, fused_channels=96)


    def forward(self, x):
        x_b = self.basis_block(x)
        x_1 = self.res_block_1(x_b)
        x_2 = self.res_block_2(x_1)
        x_3 = self.res_block_3(x_2)
        x_4 = self.res_block_4(x_3)
        x_34 = self.fpn_block_1(x_4, x_3)
        x_234 = self.fpn_block_2(x_34, x_2)

        x_234 = F.interpolate(x_234, size=(540,960), mode='bilinear', align_corners=False)

        return x_234

################## FUSION AFTER DECODER ####################
class cameraradar_fusion_Afterdecoder(nn.Module):
    def __init__(self, channels_bev, blocks,detection_head,segmentation_head,config):
        super(cameraradar_fusion_Afterdecoder, self).__init__()

        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        self.cameraonly = fftradnet_adapted(channels_bev, blocks)
        self.radaronly = PIXOR()

        if (self.detection_head):
            self.detection_header = DetectionHeader(config=config,n_input=128, n_output=128)

        if(self.segmentation_head):
            self.freespace = nn.Sequential(BasicBlock_UpScaling(128,64),
                                           BasicBlock_UpScaling(64, 32),
                                           nn.Conv2d(32, 1, kernel_size=1))

    def forward(self, cam_inputs, radar_inputs):

        out = {'Detection': [], 'Segmentation': []}
        encdec_output = self.cameraonly(cam_inputs) #result after decoder from camera decoder
        x_234 = self.radaronly(radar_inputs) #result after decoder from radar decoder

        # here we fuse (concatenate) camera and radar data
        ad_fusion = torch.cat([x_234, encdec_output], dim=1)

        if (self.detection_head):
            out['Detection'] = self.detection_header(ad_fusion)

        if (self.segmentation_head):
            frespace_pred = self.freespace(ad_fusion)
            out['Segmentation'] = frespace_pred[:, :, :, :850]

        return out

# if __name__ == "__main__":
#     camera_input = torch.randn((2, 3, 540, 960))
#     radar_input = torch.randn((2, 46, 1030, 800))
#
#     net = cameraradar_fusion_Afterdecoder(channels_bev=[16, 32, 64, 128],
#                                           blocks=[3, 6, 6, 3],
#                                           detection_head=True,segmentation_head=True)
#
#     fusion_afterdecoder = net(camera_input, radar_input)
