import torch
import torch.nn as nn
import torch.nn.functional as F

BN = torch.nn.BatchNorm2d
Conv2d = torch.nn.Conv2d

__all__ = [
    'scalablelayer', 'convlayer', 'selayer', 'gatinglayer', 'attentionlayer',
    'crossconvlayer', 'crossconvhrnetlayer'
]


class Scalable_Layer(nn.Module):
    def __init__(self,
                 channel=None,
                 channel_wise=False,
                 element_wise=False,
                 **kwargs):
        super(Scalable_Layer, self).__init__()
        self.channel = channel
        self.channel_wise = channel_wise
        self.element_wise = element_wise
        if channel_wise:
            self.w = torch.nn.Parameter(torch.FloatTensor((channel)),
                                        requires_grad=True)
        elif element_wise:
            pass
        else:
            self.w = torch.nn.Parameter(torch.FloatTensor(1),
                                        requires_grad=True)
        self.w.data.fill_(0.00)

    def forward(self, x, y):
        if self.channel_wise:
            return self.w.view(1, self.channel, 1, 1) * x
        return self.w * x


class Conv_Layer(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 **kwargs):
        super(Conv_Layer, self).__init__()
        self.conv1 = Conv2d(channel,
                            channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.bn1 = BN(channel)
        self.relu = nn.ReLU(inplace=True)
        self.use_pooling = use_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, y):
        if not self.use_pooling:
            return self.relu(self.bn1(self.conv1(x)))
        else:
            return self.avg_pool(self.relu(self.bn1(self.conv1(x))))


class Cross_Conv_Layer(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None):
        super(Cross_Conv_Layer, self).__init__()
        self.name = name
        self.layer2channel = layer2channel
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]
        self.convs = torch.nn.ModuleDict()
        self.hidden_feature_channel = 256
        self.out_channel = self.layer2channel[self.name]

        for aux_layer in self.aux_layers:
            src_channel = self.layer2channel[aux_layer]
            self.convs[aux_layer] = torch.nn.Sequential(
                Conv2d(src_channel,
                       self.hidden_feature_channel,
                       kernel_size=kernel_size,
                       stride=stride,
                       padding=padding), BN(self.hidden_feature_channel),
                nn.ReLU(inplace=True))
        self.agg_conv = torch.nn.Sequential(
            Conv2d(self.hidden_feature_channel * len(self.aux_layers),
                   self.out_channel,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding), BN(self.out_channel),
            nn.ReLU(inplace=True))

        self.use_pooling = use_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: dict, y: torch.tensor, detach=False):

        self_spatial_size = y.size()[-2:]
        transformed_aux_features = []
        for aux_layer_name in sorted(self.aux_layers):
            interpolated = F.interpolate(
                (x[aux_layer_name].detach() if detach else x[aux_layer_name]),
                size=self_spatial_size,
                mode='bilinear',
                align_corners=False)
            transformed_aux_features.append(
                self.convs[aux_layer_name](interpolated))
        transformed_aux_features = torch.cat(transformed_aux_features, dim=1)
        if self.use_pooling:
            out = self.avg_pool(self.agg_conv(transformed_aux_features))
        else:
            out = self.agg_conv(transformed_aux_features)
        return out


class Cross_Conv_HRNet_Layer(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 name=None,
                 layer2channel=None,
                 layer2auxlayers=None):
        super(Cross_Conv_HRNet_Layer, self).__init__()
        self.name = name
        self.layer2channel = layer2channel
        self.layer2auxlayers = layer2auxlayers
        self.aux_layers = self.layer2auxlayers[name]
        self.convs = torch.nn.ModuleDict()
        self.out_channel = self.layer2channel[self.name]
        self.channel_to_spatial_ratio = {
            # For ResNet
            256: 4,
            512: 2,
            1024: 1,
            # For FMTB
            768: 1,
            192: 1,
            # For MTB
            64: 4,
            128: 2,
            320: 1,
        }

        for aux_layer in self.aux_layers:
            src_channel = self.layer2channel[aux_layer]
            src_spatial_ratio = self.channel_to_spatial_ratio[src_channel]
            target_spatial_ratio = self.channel_to_spatial_ratio[
                self.out_channel]

            if target_spatial_ratio >= src_spatial_ratio:
                t = nn.Sequential(
                    nn.Conv2d(src_channel,
                              self.out_channel,
                              kernel_size=(1, 1),
                              bias=False),
                    BN(self.out_channel),
                    nn.Upsample(scale_factor=target_spatial_ratio //
                                src_spatial_ratio,
                                mode='nearest',
                                align_corners=None),
                )

            else:
                conv_layers = []
                conv_args = {
                    'kernel_size': (3, 3),
                    'stride': (2, 2),
                    'padding': 1,
                    'bias': False
                }
                for _ in range(src_spatial_ratio // target_spatial_ratio // 2 -
                               1):
                    conv_layers.append(
                        nn.Conv2d(src_channel, src_channel, **conv_args))
                    conv_layers.append(BN(src_channel))
                    conv_layers.append(nn.ReLU(False))
                conv_layers.append(
                    nn.Conv2d(src_channel, self.out_channel, **conv_args))
                conv_layers.append(BN(self.out_channel))

                t = nn.Sequential(*conv_layers)

            self.convs[aux_layer] = t

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal(m.weight, mean=0.0, std=1e-3)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: dict, y: torch.tensor, detach=False):

        transformed_aux_features = []
        for aux_layer_name in sorted(self.aux_layers):
            transformed_aux_features.append(self.convs[aux_layer_name](
                (x[aux_layer_name].detach() if detach else x[aux_layer_name])))

        out = sum(transformed_aux_features)

        return out


class Gating_Layer(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 **kwargs):
        super(Gating_Layer, self).__init__()
        self.conv1 = Conv2d(channel,
                            channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.bn1 = BN(channel)
        self.relu = nn.ReLU(inplace=True)
        self.use_pooling = use_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, y):
        if not self.use_pooling:
            return self.relu(self.bn1(self.conv1(x))) * y
        else:
            return self.avg_pool(self.relu(self.bn1(self.conv1(x)))) * y


class Attention_Layer(nn.Module):
    def __init__(self,
                 channel,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 use_pooling=False,
                 use_cross=False,
                 **kwargs):
        super(Attention_Layer, self).__init__()
        self.conv1 = Conv2d(channel,
                            channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.bn1 = BN(channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = Conv2d(channel,
                            channel,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding)
        self.bn2 = BN(channel)

        self.tanh = nn.Tanh()

        self.use_pooling = use_pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.use_cross = use_cross

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=0.001, b=0.01)
            elif isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, a=0.001, b=0.01)

    def forward(self, x, y):
        if self.use_cross:
            return self.tanh(
                self.relu(self.bn1(self.conv1(x))) *
                self.relu(self.bn2(self.conv2(y)))) + self.tanh(
                    self.avg_pool(self.relu(self.bn1(self.conv1(x)))) *
                    self.avg_pool(self.relu(self.bn2(self.conv2(y)))))
        if not self.use_pooling:
            return self.tanh(
                self.relu(self.bn1(self.conv1(x))) *
                self.relu(self.bn2(self.conv2(y))))
        else:
            return self.tanh(
                self.avg_pool(self.relu(self.bn1(self.conv1(x)))) *
                self.avg_pool(self.relu(self.bn2(self.conv2(y)))))


class MATNLayer(nn.Module):
    def __init__(self, channel, kernel_size=1, stride=1, padding=0, **kwargs):
        super(MATNLayer, self).__init__()
        self.conv1 = Conv2d(2 * channel,
                            channel,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.bn1 = BN(channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = Conv2d(channel,
                            channel,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        self.bn2 = BN(channel)
        self.sigmoid = nn.Sigmoid()

        self.conv3 = Conv2d(channel,
                            channel,
                            kernel_size=3,
                            stride=1,
                            padding=1)
        self.bn3 = BN(channel)

        init_weights = [0.9, 0.1]
        self.conv1.weight = nn.Parameter(
            torch.cat([
                torch.eye(channel) * init_weights[0],
                torch.eye(channel) * init_weights[1]
            ],
                      dim=1).view(channel, -1, 1, 1))
        # if self.conv1.bias:
        self.conv1.bias.data.fill_(0)
        self.conv2.weight = nn.Parameter(
            torch.cat([torch.eye(channel) * init_weights[0]],
                      dim=1).view(channel, -1, 1, 1))
        # if self.conv2.bias:
        self.conv2.bias.data.fill_(0)

    def forward(self, x, y):
        y = torch.cat([x, y], dim=1)
        y = self.sigmoid(
            self.bn2(self.conv2(self.relu(self.bn1(self.conv1(y))))))
        x = x * y
        return self.relu(self.bn3(self.conv3(x)))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16, **kwargs):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform(m.weight, a=0.0001, b=0.00001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.uniform(m.weight, a=0.0001, b=0.00001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, _):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SELayerPolicy(nn.Module):
    def __init__(self, channel, reduction=16, **kwargs):
        super(SELayerPolicy, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False), nn.Sigmoid())

    def forward(self, x, _):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def scalablelayer(channel, **kwargs):
    return Scalable_Layer(channel, **kwargs)


def convlayer(channel, **kwargs):
    return Conv_Layer(channel, **kwargs)


def selayer(channel, **kwargs):
    return SELayer(channel, **kwargs)


def gatinglayer(channel, **kwargs):
    return Gating_Layer(channel, **kwargs)


def attentionlayer(channel, **kwargs):
    return Attention_Layer(channel, **kwargs)


def crossconvlayer(channel, **kwargs):
    return Cross_Conv_Layer(channel, **kwargs)


def crossconvhrnetlayer(channel, **kwargs):
    return Cross_Conv_HRNet_Layer(channel, **kwargs)
