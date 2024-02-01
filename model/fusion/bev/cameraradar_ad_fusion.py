import torch
import torch.nn as nn
import torch.nn.functional as F

NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

class Detection_Header(nn.Module):

    def __init__(self,config, use_bn=True, reg_layer=2):
        super(Detection_Header, self).__init__()

        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.config = config
        bias = not use_bn

        if config['model']['DetectionHead']=='True':
            self.conv1 = conv3x3(384, 144, bias=bias)
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)

            self.conv3 = conv3x3(96, 96, bias=bias)
            self.bn3 = nn.BatchNorm2d(96)
            self.conv4 = conv3x3(96, 96, bias=bias)
            self.bn4 = nn.BatchNorm2d(96)

            self.clshead = conv3x3(96, 1, bias=True)
            self.reghead = conv3x3(96, reg_layer, bias=True)

    def forward(self, x):

        if self.config['model']['DetectionHead']=='True':
            x = self.conv1(x)
            if self.use_bn:
                x = self.bn1(x)
            x = self.conv2(x)
            if self.use_bn:
                x = self.bn2(x)
            x = self.conv3(x)
            if self.use_bn:
                x = self.bn3(x)
            x = self.conv4(x)
            if self.use_bn:
                x = self.bn4(x)

            cls = torch.sigmoid(self.clshead(x))
            reg = self.reghead(x)

            return torch.cat([cls, reg], dim=1)


class Bottleneck(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None, expansion=4):
        super(Bottleneck, self).__init__()
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


class MIMO_PreEncoder(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size=(1, 12), dilation=(1, 16), use_bn=False):
        super(MIMO_PreEncoder, self).__init__()
        self.use_bn = use_bn

        self.conv = nn.Conv2d(in_layer, out_layer, kernel_size,
                              stride=(1, 1), padding=0, dilation=dilation, bias=(not use_bn))

        self.bn = nn.BatchNorm2d(out_layer)
        self.padding = int(NbVirtualAntenna / 2)

    def forward(self, x):
        width = x.shape[-1]
        x = torch.cat([x[..., -self.padding:], x, x[..., :self.padding]], axis=3)
        x = self.conv(x)
        x = x[..., int(x.shape[-1] / 2 - width / 2):int(x.shape[-1] / 2 + width / 2)]

        if self.use_bn:
            x = self.bn(x)
        return x

class FPN_BackBone_radar(nn.Module):

    def __init__(self, num_block, channels, block_expansion, mimo_layer, use_bn=True):
        super(FPN_BackBone_radar, self).__init__()

        self.block_expansion = block_expansion
        self.use_bn = use_bn

        # pre processing block to reorganize MIMO channels
        self.pre_enc = MIMO_PreEncoder(32, mimo_layer,
                                       kernel_size=(1, NbTxAntenna),
                                       dilation=(1, NbRxAntenna),
                                       use_bn=True)

        self.in_planes = mimo_layer

        self.conv = conv3x3(self.in_planes, self.in_planes)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=False)

        # Residuall blocks
        self.block1 = self._make_layer(Bottleneck, planes=channels[0], num_blocks=num_block[0])
        self.block2 = self._make_layer(Bottleneck, planes=channels[1], num_blocks=num_block[1])
        self.block3 = self._make_layer(Bottleneck, planes=channels[2], num_blocks=num_block[2])
        self.block4 = self._make_layer(Bottleneck, planes=channels[3], num_blocks=num_block[3])

    def forward(self, x):

        x = self.pre_enc(x)
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
        return nn.Sequential(*layers)


class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out


class RangeAngle_Decoder(nn.Module):
    def __init__(self, ):
        super(RangeAngle_Decoder, self).__init__()

        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0))

        self.conv_block4 = BasicBlock(48, 128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2, 1), padding=1, output_padding=(1, 0))
        self.conv_block3 = BasicBlock(192, 256)

        self.L3 = nn.Conv2d(192, 224, kernel_size=1, stride=1, padding=0)
        self.L2 = nn.Conv2d(160, 224, kernel_size=1, stride=1, padding=0)

    def forward(self, features):
        T4 = features['x4'].transpose(1, 3)
        T3 = self.L3(features['x3']).transpose(1, 3)
        T2 = self.L2(features['x2']).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4), T3), axis=1)
        S4 = self.conv_block4(S4)

        S43 = torch.cat((self.deconv3(S4), T2), axis=1)
        out = self.conv_block3(S43) #.detach().clone()

        return out


class PolarSegFusionNet_RA(nn.Module):
    def __init__(self, mimo_layer, channels, blocks):
        super(PolarSegFusionNet_RA, self).__init__()

        self.FPN = FPN_BackBone_radar(num_block=blocks, channels=channels, block_expansion=4, mimo_layer=mimo_layer,
                                      use_bn=True)

        self.RA_decoder = RangeAngle_Decoder()

    def forward(self, x):
        features = self.FPN(x)

        RA_decoded = self.RA_decoder(features)

        return RA_decoded


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

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class BEV_decoder(nn.Module):
    def __init__(self, ):
        super(BEV_decoder, self).__init__()
        self.up1 = (UpScaling(512, 256))
        self.up2 = (UpScaling(256, 128))
        self.M1 = nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, features):
        T2 = features['x2']
        T3 = features['x3']
        T4 = features['x4']

        x = self.up1(T4, T3)
        x = self.up2(x, T2)
        x = x.transpose(1, 3)
        out = self.M1(x).transpose(1, 3)

        width = out.shape[-1]
        out2 = out
        out = out2[:, :, 0:128, 16:int(width - 16)]
        return out

class PolarSegFusionNet_BEV(nn.Module):
    def __init__(self, channels_bev, blocks):

        super(PolarSegFusionNet_BEV, self).__init__()

        self.FPN = FPN_BackBone_camera(num_block=blocks, channels=channels_bev, block_expansion=4, use_bn=True)
        self.BEV_decoder = BEV_decoder()


    def forward(self, x):
        features = self.FPN(x)
        BEV_decoded = self.BEV_decoder(features)
        return BEV_decoded

##################################
############FUSION################
##################################

class cameraradar_fusion_Afterdecoder_bev(nn.Module):
    def __init__(self, mimo_layer, channels, channels_bev, blocks, detection_head, segmentation_head, config, regression_layer=2):
        super(cameraradar_fusion_Afterdecoder_bev, self).__init__()

        self.detection_head = detection_head
        self.segmentation_head = segmentation_head

        self.radarenc = PolarSegFusionNet_RA(mimo_layer=mimo_layer, channels=channels, blocks=blocks)
        self.cameraenc = PolarSegFusionNet_BEV(channels_bev=channels_bev, blocks=blocks)
        if (self.detection_head):
            self.detection_header = Detection_Header(config=config, reg_layer=regression_layer)
        if (self.segmentation_head):
            self.freespace = nn.Sequential(BasicBlock(384, 128), BasicBlock(128, 64), nn.Conv2d(64, 1, kernel_size=1))

    def forward(self, bev_inputs, ra_inputs):

        out = {'Detection':[],'Segmentation':[]}

        RA_decoded = self.radarenc(ra_inputs)

        BEV_decoded = self.cameraenc(bev_inputs)

        out_fused = torch.cat((RA_decoded, BEV_decoded), axis=1)

        if (self.detection_head):
            out['Detection'] = self.detection_header(out_fused)

        if (self.segmentation_head):
            Y = F.interpolate(out_fused, (256, 224))
            out['Segmentation'] = self.freespace(Y)

        return out

# if __name__ == "__main__":
#
#     #RD params
#     mimo_layer = 192
#     channels = [32, 40, 48, 56]
#     blocks = [3, 6, 6, 3]
#
#     #BEV params
#     channels_bev = [16, 32, 64, 128]
#
#     #inputs
#     ra_inputs = torch.randn((2, 32, 512, 256))
#     bev_inputs = torch.randn((2, 3, 512, 256))
#
#     #fusion
#     net = cameraradar_fusion_Afterdecoder_bev(mimo_layer=mimo_layer, channels=channels,
#                                              channels_bev=channels_bev, blocks=blocks,
#                                              detection_head=True, segmentation_head=True,
#                                              regression_layer=2)
#     #print(net)
#     fused_arch = net(ra_inputs, bev_inputs)