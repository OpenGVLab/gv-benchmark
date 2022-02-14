import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm
from mmseg.models.builder import BACKBONES
from timm.models.layers import DropPath, Mlp, lecun_normal_, trunc_normal_
from torch.nn.init import zeros_

# will be overwritten during runtime
BN = nn.SyncBatchNorm
Conv2d = nn.Conv2d

__all__ = [
    'to3d', 'to4d', 'SqueezeExcitation', 'FusedMBConv3x3', 'MBConv3x3',
    'SABlock', 'Block'
]


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def to4d(x, h, w):
    if len(x.shape) == 4:
        return x
    B, N, C = x.shape
    return x.transpose(1, 2).reshape(B, C, h, w)


def to3d(x):
    if len(x.shape) == 3:
        return x
    B, C, h, w = x.shape
    N = h * w
    return x.reshape(B, C, N).transpose(1, 2)


def hard_sigmoid(x, inplace=False):
    return F.relu6(x + 3, inplace) / 6


class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = _make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)

    def _scale(self, input, inplace):
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        # return F.hardsigmoid(scale, inplace=inplace)
        return hard_sigmoid(scale, inplace=inplace)

    def forward(self, input):
        scale = self._scale(input, True)
        return scale * input


class FusedMBConv3x3(nn.Module):
    def __init__(self,
                 in_features,
                 out_features=None,
                 stride=1,
                 mlp_ratio=4,
                 use_se=True,
                 drop=0.,
                 drop_path=0.,
                 seq_l=196,
                 head_dim=32,
                 **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        if in_features != out_features or stride != 1:
            self.residual = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_features, out_features, kernel_size=1))
        else:
            self.residual = nn.Identity()

        self.b1 = None
        if in_features != hidden_features:
            layers_b1 = []
            layers_b1.append(BN(in_features))
            layers_b1.append(
                Conv2d(in_features,
                       hidden_features,
                       kernel_size=(3, 3),
                       stride=stride,
                       padding=(1, 1),
                       bias=False))
            layers_b1.append(BN(hidden_features))
            layers_b1.append(nn.GELU())
            self.b1 = nn.Sequential(*layers_b1)

        layers = []
        if use_se:
            layers.append(SqueezeExcitation(hidden_features))

        if in_features != hidden_features:
            kernel_size = 1
            padding = 0
            stride = 1
        else:
            kernel_size = 3
            padding = 1
            stride = stride
        layers.append(
            Conv2d(hidden_features,
                   out_features,
                   kernel_size=kernel_size,
                   padding=padding,
                   stride=stride))
        layers.append(BN(out_features))
        self.b2 = nn.Sequential(*layers)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        zeros_(self.b2[-1].weight)

    def forward(self, x, h=14, w=14):
        h = int(h)
        w = int(w)
        residual = self.residual(x)
        if self.b1 is not None:
            x = self.b1(x)
        x = self.b2(x)

        return residual + self.drop_path(x)


class MBConv3x3(nn.Module):
    def __init__(self,
                 in_features,
                 out_features=None,
                 stride=1,
                 mlp_ratio=4,
                 use_se=True,
                 drop=0.,
                 drop_path=0.,
                 seq_l=196,
                 head_dim=32,
                 **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        if in_features != out_features or stride != 1:
            self.residual = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_features, out_features, kernel_size=1))
        else:
            self.residual = nn.Identity()

        self.b1 = None
        if in_features != hidden_features or stride != 1:
            layers_b1 = []
            layers_b1.append(BN(in_features))
            layers_b1.append(
                Conv2d(in_features,
                       hidden_features,
                       kernel_size=(1, 1),
                       stride=1,
                       padding=(0, 0),
                       bias=False))
            layers_b1.append(BN(hidden_features))
            layers_b1.append(nn.GELU())
            self.b1 = nn.Sequential(*layers_b1)

        layers = []
        layers.append(
            Conv2d(hidden_features,
                   hidden_features,
                   kernel_size=(3, 3),
                   padding=(1, 1),
                   groups=hidden_features,
                   stride=stride,
                   bias=False))
        layers.append(BN(hidden_features))
        layers.append(nn.GELU())
        if use_se:
            layers.append(SqueezeExcitation(hidden_features))

        layers.append(
            Conv2d(hidden_features,
                   out_features,
                   kernel_size=(1, 1),
                   padding=(0, 0)))
        layers.append(BN(out_features))
        self.b2 = nn.Sequential(*layers)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        zeros_(self.b2[-1].weight)

    def forward(self, x, h=14, w=14):
        h = int(h)
        w = int(w)
        residual = self.residual(x)
        if self.b1 is not None:
            x = self.b1(x)
        x = self.b2(x)

        return residual + self.drop_path(x)


class OldAttention(nn.Module):
    def __init__(self,
                 dim,
                 out_dim=None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 static=False,
                 seq_l=196):
        super().__init__()
        out_dim = out_dim or dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.static = static
        if self.static:
            self.static_a = nn.Parameter(
                torch.Tensor(1, num_heads, seq_l, seq_l))
            trunc_normal_(self.static_a)
        self.custom_flops = 2 * seq_l * seq_l * dim

    def forward(self, x, head=0, mask_type=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        if mask_type:
            mask = torch.ones_like(qkv)
            mask[:, :, head] = 0
            if mask_type == 'layer':
                mask = 1 - mask
            qkv = qkv * mask
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.static:
            attn = attn + self.static_a
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 out_dim=None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 static=False,
                 seq_l=196):
        super().__init__()
        out_dim = out_dim or dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.static = static
        if self.static:
            self.static_a = nn.Parameter(
                torch.Tensor(1, num_heads, seq_l, seq_l))
            trunc_normal_(self.static_a)
        self.custom_flops = 2 * seq_l * seq_l * dim

    def forward(self, x, head=0, mask_type=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float(
        )  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.static:
            attn = attn + self.static_a
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        if qkv.dtype == torch.float16:
            x = x.half()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class OldXCA(nn.Module):
    """Cross-Covariance Attention (XCA) operation where the channels are
    updated using a weighted sum.

    The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """
    def __init__(self,
                 dim,
                 out_dim=None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 static=False,
                 seq_l=196):
        super().__init__()
        out_dim = out_dim or dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

    def forward(self, x, head=0, mask_type=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class XCA(nn.Module):
    """Cross-Covariance Attention (XCA) operation where the channels are
    updated using a weighted sum.

    The weights are obtained from the (softmax normalized) Cross-covariance
    matrix (Q^T K \\in d_h \\times d_h)
    """
    def __init__(self,
                 dim,
                 out_dim=None,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 static=False,
                 seq_l=196):
        super().__init__()
        out_dim = out_dim or dim
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

    def forward(self, x, head=0, mask_type=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float(
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)

        if qkv.dtype == torch.float16:
            x = x.half()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SABlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 stride=1,
                 mlp_ratio=4,
                 head_dim=32,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 drop_path=0.,
                 attn_drop=0.,
                 seq_l=196,
                 conv_embedding=1,
                 init_values=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.stride = stride
        self.in_features = in_features
        self.out_features = out_features
        mlp_hidden_dim = int(in_features * mlp_ratio)
        num_heads = in_features // head_dim
        self.init_values = init_values
        self.conv_embedding = conv_embedding
        if conv_embedding == 1:
            self.pos_embed = nn.Conv2d(in_features,
                                       in_features,
                                       3,
                                       padding=1,
                                       groups=in_features)
        elif conv_embedding == 2:
            self.pos_embed = nn.Linear(seq_l * stride**2, seq_l * stride**2)
        elif conv_embedding == 3:
            self.pos_embed = nn.MaxPool2d(3, 1, 1)
        else:
            self.pos_embed = None
        self.attn = Attention(in_features,
                              out_features,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              seq_l=seq_l)
        if stride != 1 or in_features != out_features:
            self.ds = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
            self.residual = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_features, out_features, kernel_size=1))
            if init_values != -1:
                self.gamma_1 = nn.Parameter(init_values * torch.ones(
                    (out_features)),
                                            requires_grad=True)
        else:
            self.norm2 = nn.LayerNorm(in_features)
            self.mlp = Mlp(in_features=in_features,
                           hidden_features=mlp_hidden_dim,
                           act_layer=nn.GELU,
                           drop=drop)
            if init_values != -1:
                self.gamma_1 = nn.Parameter(init_values * torch.ones(
                    (out_features)),
                                            requires_grad=True)
                self.gamma_2 = nn.Parameter(init_values * torch.ones(
                    (out_features)),
                                            requires_grad=True)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, h=14, w=14):
        h = int(h)
        w = int(w)
        if self.conv_embedding == 1 or self.conv_embedding == 3:
            x = x + to3d(self.pos_embed(to4d(x, h, w)))
        elif self.conv_embedding == 2:
            x = x + self.pos_embed(x.transpose(1, 2)).transpose(1, 2)
        if self.stride == 1 and self.in_features == self.out_features:
            if self.init_values != -1:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            residual = to3d(self.residual(to4d(x, h, w)))
            x = self.norm1(x)
            x = to3d(self.ds(to4d(x, h, w)))
            x = self.attn(x)
            # x = residual + self.drop_path(x)
            if self.init_values != -1:
                x = residual + self.gamma_1 * x
            else:
                x = residual + x
        return x


class Block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 stride=1,
                 mlp_ratio=4,
                 head_dim=32,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 drop_path=0.,
                 attn_drop=0.,
                 seq_l=196,
                 conv_embedding=1,
                 init_values=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_features)
        self.stride = stride
        self.in_features = in_features
        self.out_features = out_features
        mlp_hidden_dim = int(in_features * mlp_ratio)
        num_heads = in_features // head_dim
        self.init_values = init_values
        self.conv_embedding = conv_embedding
        if conv_embedding == 1:
            self.pos_embed = nn.Conv2d(in_features,
                                       in_features,
                                       3,
                                       padding=1,
                                       groups=in_features)
        elif conv_embedding == 2:
            self.pos_embed = nn.Linear(seq_l * stride**2, seq_l * stride**2)
        elif conv_embedding == 3:
            self.pos_embed = nn.MaxPool2d(3, 1, 1)
        else:
            self.pos_embed = None
        self.attn = XCA(in_features,
                        out_features,
                        num_heads=num_heads,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=drop,
                        seq_l=seq_l)
        if stride != 1 or in_features != out_features:
            self.ds = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
            self.residual = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_features, out_features, kernel_size=1))
            if init_values != -1:
                self.gamma_1 = nn.Parameter(init_values * torch.ones(
                    (out_features)),
                                            requires_grad=True)
        else:
            self.norm2 = nn.LayerNorm(in_features)
            self.mlp = Mlp(in_features=in_features,
                           hidden_features=mlp_hidden_dim,
                           act_layer=nn.GELU,
                           drop=drop)
            if init_values != -1:
                self.gamma_1 = nn.Parameter(init_values * torch.ones(
                    (out_features)),
                                            requires_grad=True)
                self.gamma_2 = nn.Parameter(init_values * torch.ones(
                    (out_features)),
                                            requires_grad=True)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, h=14, w=14):
        h = int(h)
        w = int(w)
        if self.conv_embedding == 1 or self.conv_embedding == 3:
            x = x + to3d(self.pos_embed(to4d(x, h, w)))
        elif self.conv_embedding == 2:
            x = x + self.pos_embed(x.transpose(1, 2)).transpose(1, 2)
        if self.stride == 1 and self.in_features == self.out_features:
            if self.init_values != -1:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            residual = to3d(self.residual(to4d(x, h, w)))
            x = self.norm1(x)
            x = to3d(self.ds(to4d(x, h, w)))
            x = self.attn(x)
            # x = residual + self.drop_path(x)
            if self.init_values != -1:
                x = residual + self.gamma_1 * x
            else:
                x = residual + x
        return x


block_op_map = {
    'MBConv3x3': MBConv3x3,
    'SABlock': SABlock,
    'Block': Block,
    'FusedMBConv3x3': FusedMBConv3x3
}


@BACKBONES.register_module()
class MetaNet(BaseModule):
    def __init__(self,
                 repeats,
                 expansion,
                 channels,
                 strides=[1, 2, 2, 2, 1, 2],
                 num_classes=1000,
                 drop_path_rate=0.,
                 input_size=224,
                 weight_init='',
                 head_dim=48,
                 final_drop=0.0,
                 init_values=1e-6,
                 conv_embedding=1,
                 block_ops=['MBConv3x3'] * 3 + ['Block'] * 3,
                 stem_dim=32,
                 final_head_dim=1280,
                 out_stages=[1, 2, 4, 5],
                 checkpoint=False,
                 conv_func=None,
                 norm_cfg=None,
                 neck_only=False,
                 init_cfg=None,
                 norm_eval=False,
                 frozen_stages=-1,
                 pretrained=None):
        super(MetaNet, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            warnings.warn('no init')
        else:
            raise TypeError('pretrained must be a str or None')

        assert not neck_only  # neck currently unsupported
        self.checkpoint = checkpoint

        global BN
        global Conv2d

        if norm_cfg is not None:
            raise NotImplementedError()

        # BN = norm_cfg
        # Conv2d = conv_func

        self.num_classes = num_classes

        self.stem = nn.Sequential(
            Conv2d(3,
                   stem_dim,
                   kernel_size=(3, 3),
                   stride=(2, 2),
                   padding=(1, 1),
                   bias=False),
            BN(stem_dim),
            nn.GELU(),
        )

        block_ops_str = block_ops
        block_ops = []
        for op in block_ops_str:
            block_ops.append(block_op_map[op])

        # repeats =   [1, 2, 2, 3, 5, 8]
        # strides =   [1, 2, 2, 2, 1, 2]
        # expansion = [1, 6, 6, 4, 4, 4]
        # channels =  [16, 32, 48, 96, 128, 192]
        print(f'drop_path_rate {drop_path_rate}', 0)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(repeats))
        ]  # stochastic depth decay rule
        dpr.reverse()

        cin = stem_dim
        blocks = []
        seq_l = (input_size // 2)**2
        self.conv_num = sum(repeats[:3])
        self.scale = []
        self.out_blocks = []
        self.output_dimensions = []
        for stage in range(len(strides)):
            cout = channels[stage]
            # block_op = MBConv3x3 if stage < 3 else Block
            block_op = block_ops[stage]
            print(
                f'stage {stage}, cin {cin}, cout {cout}, \
                    s {strides[stage]}, e {expansion[stage]} b {block_op}', 0)
            seq_l = seq_l // (strides[stage]**2)
            for i in range(repeats[stage]):
                stride = strides[stage] if i == 0 else 1
                blocks.append(
                    block_op(cin,
                             cout,
                             stride=stride,
                             mlp_ratio=expansion[stage],
                             drop_path=dpr.pop(),
                             seq_l=seq_l,
                             head_dim=head_dim,
                             init_values=init_values,
                             conv_embedding=conv_embedding))
                cin = cout
                self.scale.append(stride)
            if stage in out_stages:
                self.out_blocks.append(len(blocks) - 1)
                self.output_dimensions.append(cout)
        self.blocks = nn.Sequential(*blocks)

        head_dim = final_head_dim
        self.head = nn.Sequential(
            Conv2d(cout,
                   head_dim,
                   kernel_size=(1, 1),
                   stride=(1, 1),
                   padding=(0, 0),
                   bias=False),
            BN(head_dim),
            nn.GELU(),
        )
        self.final_drop = nn.Dropout(
            final_drop) if final_drop > 0.0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Linear(head_dim, num_classes)
        self.output_dimensions.append(head_dim)
        print('self.output_dimensions', self.out_blocks,
              self.output_dimensions)
        head_bias = -math.log(
            self.num_classes) if 'nlhb' in weight_init else 0.
        # trunc_normal_(self.pos_embed, std=.02)
        # Weight init
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)
        else:
            # trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {'pos_embed', 'cls_token', 'dist_token'}
        return {'pos_embed'}

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.stem]:
                for param in m.parameters():
                    param.requires_grad = False

        # if self.frozen_stages == 4:
        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward_features(self, x):
        x = self.stem(x)
        ret, layer = [], 0
        b, c, h, w = x.shape
        for i, blk in enumerate(self.blocks):
            if isinstance(blk, (SABlock, Block)):
                x = to3d(x)
            else:
                assert isinstance(blk, (MBConv3x3, FusedMBConv3x3))
                x = to4d(x, h, w)
            if self.checkpoint:
                x = checkpoint.checkpoint(blk, x, torch.Tensor([h]),
                                          torch.Tensor([w]))
            else:
                x = blk(x, h, w)
            h = math.ceil(h / self.scale[i])
            w = math.ceil(w / self.scale[i])
            if layer < len(self.out_blocks) and i == self.out_blocks[layer]:
                ret.append(to4d(x, h, w))
                layer += 1
        return ret

    def forward(self, x):
        ret = self.forward_features(x)
        # x = self.final_drop(x)
        # x = self.classifier(x)
        # ret['pre_fc'] = x
        return tuple(ret)

    def train(self, mode=True):
        super(MetaNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


def _init_vit_weights(m,
                      n: str = '',
                      head_bias: float = 0.,
                      jax_impl: bool = False):
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        elif n.startswith('pre_logits'):
            lecun_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if 'mlp' in n:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.zeros_(m.bias)
            else:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif jax_impl and isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
