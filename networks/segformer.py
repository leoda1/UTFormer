# -*- coding: UTF-8 -*-
"""
===================================================================================
@author : Leoda
@Date   : 2023/10/30 21:32:41
@Project -> : learn$
==================================================================================
"""
import math
import warnings
import numpy as np
import torch
import torch.nn as nn

from functools import partial
import torch.nn.functional as F

#=================================================================================
# encoder
#=================================================================================
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # 公式来源：https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # 定义正态分布累积分布函数
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # 如果 mean 值超过了截断范围 [a, b] 的 2 倍标准差，发出警告
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        # 使用截断均匀分布生成张量的值，并使用正态分布的逆累积分布函数将其转换为截断标准正态分布。
        # 获取上限和下限的累积分布值
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        # 用 [l, u] 之间的均匀分布填充张量，然后将其平移到 [2l-1, 2u-1] 范围内。
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        # 使用正态分布的逆累积分布函数将张量转换为截断标准正态分布
        tensor.erfinv_()
        # 转换为正确的均值和标准差
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        # 确保张量的值在截断范围内
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    函数用于在给定的 Tensor 中填充值，这些值是从一个截断正态分布中随机抽取的。
    截断正态分布与标准的正态分布类似，但是在指定的范围 [a, b] 之外的值会被重新绘制，
    直到它们位于范围内。这个函数是 PyTorch 中用于初始化权重的方法之一，可以在神经网
    络的权重初始化阶段使用。
    Args:
        tensor: 需要填充随机值的 n 维 PyTorch 张量。
        mean: 截断正态分布的均值。默认值为 0。
        std: 截断正态分布的标准差。默认值为 1。
        a: 最小截断值。如果从截断正态分布中随机抽取的值小于 a，则会重新绘制直到它们在范围内。默认值为 -2。
        b: 最大截断值。如果从截断正态分布中随机抽取的值大于 b，则会重新绘制直到它们在范围内。默认值为 2。
    Examples:
        import torch
        import torch.nn as nn
        w = torch.empty(3, 5)  # 创建一个大小为 (3, 5) 的空张量
        nn.init.trunc_normal_(w)  # 使用截断正态分布填充张量 w
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# copy from torch.nn.functional
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # 前向传播函数
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


# --------------------------------------------------------------------------------------------------------------------#
#   Attention机制
#   将输入的特征qkv特征进行划分，首先生成query, key, value。query是查询向量、key是键向量、v是值向量。
#   然后利用 查询向量query 叉乘 转置后的键向量key，这一步可以通俗的理解为，利用查询向量去查询序列的特征，获得序列每个部分的重要程度score。
#   然后利用 score 叉乘 value，这一步可以通俗的理解为，将序列每个部分的重要程度重新施加到序列的值上去。
#
#   在segformer中，为了减少计算量，首先对特征图进行了浓缩，所有特征层都压缩到原图的1/32。
#   当输入图片为512, 512时，Block1的特征图为128, 128，此时就先将特征层压缩为16, 16。
#   在Block1的Attention模块中，相当于将8x8个特征点进行特征浓缩，浓缩为一个特征点。
#   然后利用128x128个查询向量对16x16个键向量与值向量进行查询。尽管键向量与值向量的数量较少，但因为查询向量的不同，依然可以获得不同的输出。
# --------------------------------------------------------------------------------------------------------------------#
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        # 断言dim必须能够被num_heads整除，确保每个头的维度相等
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim  # 输入维度
        self.num_heads = num_heads  # 头的数量
        head_dim = dim // num_heads  # 每个头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子，用于调整注意力分数

        self.q = nn.Linear(dim, dim, bias=qkv_bias)  # Q线性层

        self.sr_ratio = sr_ratio  # 空间缩减比率
        if sr_ratio > 1:  # 如果空间缩减比率大于1，则定义卷积层和层归一化
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)  # K和V线性层

        self.attn_drop = nn.Dropout(attn_drop)  # 注意力dropout层

        self.proj = nn.Linear(dim, dim)  # 投影线性层
        self.proj_drop = nn.Dropout(proj_drop)  # 投影dropout层

        self.apply(self._init_weights)  # 初始化权重

    def _init_weights(self, m):  # 初始化权重的辅助函数
        if isinstance(m, nn.Linear):  # 对于线性层的初始化
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):  # 对于层归一化的初始化
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):  # 何凯明初始化方法
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape  # 输入的形状
        # bs, 16384, 32 => bs, 16384, 32 => bs, 16384, 8, 4 => bs, 8, 16384, 4
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 计算Q

        if self.sr_ratio > 1:  # 如果空间缩减比率大于1，则进行卷积和层归一化处理
            # bs, 16384, 32 => bs, 32, 128, 128
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # bs, 32, 128, 128 => bs, 32, 16, 16 => bs, 256, 32
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            # bs, 256, 32 => bs, 256, 64 => bs, 256, 2, 8, 4 => 2, bs, 8, 256, 4
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # 获取K和V

        # bs, 8, 16384, 4 @ bs, 8, 4, 256 => bs, 8, 16384, 256
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 计算注意力分数

        attn = attn.softmax(dim=-1)  # 对注意力分数进行softmax
        attn = self.attn_drop(attn)  # 对注意力进行dropout处理

        # bs, 8, 16384, 256  @ bs, 8, 256, 4 => bs, 8, 16384, 4 => bs, 16384, 32
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 计算加权和
        # bs, 16384, 32 => bs, 16384, 32
        x = self.proj(x)  # 通过投影线性层
        x = self.proj_drop(x)  # 对投影进行dropout处理

        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """
    按样本丢弃路径（随机深度），当应用于残差块的主路径时。
    随机深度，通过以给定的概率随机地"丢弃"某些路径（即将它们设置为0）来增加训练过程中的随机性。这种方法有助于防止过
    拟合，并可能增加训练速度。当drop_prob设为0时，这个函数就不做任何事情，直接返回输入张量。当scale_by_keep设
    为True时，保持的路径将按保持概率进行缩放，以确保整体的期望值保持不变。
    """
    if drop_prob == 0. or not training:  # 如果drop概率为0或不在训练模式，则不做任何操作
        return x
    keep_prob = 1 - drop_prob  # 计算保持概率
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 适用于不同维度的张量，而不仅是2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)  # 用保持概率生成伯努利分布的随机张量
    if keep_prob > 0.0 and scale_by_keep:  # 如果需要按保持概率进行缩放
        random_tensor.div_(keep_prob)  # 则将随机张量除以保持概率
    return x * random_tensor  # 返回元素乘以随机张量的结果，部分路径将被随机丢弃


class DropPath(nn.Module):  # DropPath类封装drop_path函数，将其作为一个PyTorch模块，方便地集成到深度学习模型中
    def __init__(self, drop_prob=None, scale_by_keep=True):  # 构造函数，接受丢弃概率和是否按保持概率进行缩放作为参数
        super(DropPath, self).__init__()  # 调用父类的构造函数
        self.drop_prob = drop_prob  # 将丢弃概率保存为成员变量
        self.scale_by_keep = scale_by_keep  # 将是否按保持概率缩放保存为成员变量

    def forward(self, x):  # 定义前向传播函数
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)  # 调用drop_path函数，并传入保存的参数和输入x


class DWConv(nn.Module):  # 定义一个名为DWConv的类，继承自PyTorch的基础模块类
    """
    DWConv类实现了深度可分离卷积操作。深度可分离卷积是一种有效的卷积方法，它分两步进行：首先在每个输入通道上
    独立地应用卷积，然后通过1x1的卷积组合这些通道。与标准卷积相比，深度可分离卷积减少了计算和参数数量，因此可
    以提高效率。
    在本类中，卷积层的组数等于输入通道数，因此每个通道的卷积是独立的。输入的形状应当是(B, N, C)，其中B是批
    大小，N是空间维度的乘积，C是通道数。通过H和W参数，输入被重新塑形为形状(B, C, H, W)的4D张量，然后应用
    卷积。最终的输出与输入具有相同的形状。
    """

    def __init__(self, dim=768):  # 构造函数，接收一个参数dim，默认值为768
        super(DWConv, self).__init__()  # 调用父类的构造函数
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)  # 创建一个深度可分离卷积层，卷积核大小为3，步长为1，填充为1

    def forward(self, x, H, W):  # 定义前向传播函数，接收输入张量x和两个维度H和W
        B, N, C = x.shape  # 获取输入张量的形状
        x = x.transpose(1, 2).view(B, C, H, W)  # 调整张量的形状，使其适合卷积操作
        x = self.dwconv(x)  # 通过深度可分离卷积层处理张量
        x = x.flatten(2).transpose(1, 2)  # 将卷积后的张量展平并转置，以恢复原始形状

        return x  # 返回处理后的张量


class Mlp(nn.Module):  # 定义一个名为Mlp的类，继承自PyTorch的基础模块类
    """
    Mlp类定义了一个具有深度可分离卷积层的多层感知器结构。该结构首先通过一个全连接层，然后通过深度可分离卷积层,
    接着是激活函数层和Dropout层，最后通过另一个全连接层和Dropout层。全连接层可以捕捉输入特征之间的线性关系,
    深度可分离卷积层可以捕捉空间结构信息，激活函数增加了非线性，Dropout层有助于防止过拟合。这种结合了卷积和
    全连接层的设计可以更好地适应具有空间结构的输入数据。
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):  # 构造函数
        super().__init__()  # 调用父类的构造函数
        out_features = out_features or in_features  # 如果未指定输出特征数，则与输入特征数相同
        hidden_features = hidden_features or in_features  # 如果未指定隐藏层特征数，则与输入特征数相同

        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一个全连接层
        self.dwconv = DWConv(hidden_features)  # 深度可分离卷积层
        self.act = act_layer()  # 激活函数层，使用GELU或其他指定的激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二个全连接层
        self.drop = nn.Dropout(drop)  # Dropout层，用于防止过拟合

        self.apply(self._init_weights)  # 应用权重初始化方法

    def _init_weights(self, m):  # 定义权重初始化方法
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):  # 定义前向传播函数，接收输入张量x和两个维度H和W
        x = self.fc1(x)  # 通过第一个全连接层
        x = self.dwconv(x, H, W)  # 通过深度可分离卷积层
        x = self.act(x)  # 通过激活函数层
        x = self.drop(x)  # 通过Dropout层
        x = self.fc2(x)  # 通过第二个全连接层
        x = self.drop(x)  # 再次通过Dropout层
        return x  # 返回输出张量


class Block(nn.Module):  # 定义一个名为Block的类，继承自PyTorch的基础模块类
    """
    Block类定义了一个Transformer编码器块的结构。每个块包括两个主要部分：一个自注意力层和一个多层感知器（MLP）。
    这两部分都有相应的层归一化和DropPath。
    自注意力层用于捕捉输入序列中不同位置之间的依赖关系，MLP层增加了模型的非线性表示能力。层归一化有助于训练过程的
    稳定性，DropPath则是一种正则化技术。
    这种块结构是Transformer网络的基础构建块，并且可以通过堆叠多个块来构建整个Transformer网络。
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm, sr_ratio=1):  # 构造函数，接收多个参数
        super().__init__()  # 调用父类的构造函数
        self.norm1 = norm_layer(dim)  # 第一个层归一化，通常是LayerNorm
        self.attn = Attention(  # 多头自注意力层，其中包括了可缩放的点积自注意力
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
        )
        self.norm2 = norm_layer(dim)  # 第二个层归一化
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)  # MLP层

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # DropPath层，用于随机丢弃路径

        self.apply(self._init_weights)  # 应用权重初始化方法

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):  # 定义前向传播函数
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # 输入经过层归一化、自注意力和DropPath后与原始x相加
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))  # 输入经过第二个层归一化、MLP和DropPath后与上一步的结果相加
        return x  # 返回输出张量


# ----------------------------------#
#   Transformer模块，共有四个部分
# ----------------------------------#
class MixVisionTransformer(nn.Module):
    """
    通过不同大小的分块、不同层数的Transformer块来构建了一个混合视觉Transformer网络，对图像进行特征提取。
    其设计能够捕获不同尺度和层次的特征，有助于更精确地进行图像分类任务
    """

    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes  # 类别数，用于分类任务
        self.depths = depths  # 每个阶段的 Transformer Block 的重复次数

        # 设置每个 Transformer Block 中 DropPath 的概率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ----------------------------------#
        #   block1
        # ----------------------------------#
        # -----------------------------------------------#
        #   对输入图像进行分区，并下采样
        #   512, 512, 3 => 128, 128, 32 => 16384, 32
        # -----------------------------------------------#
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        # -----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   16384, 32 => 16384, 32
        # -----------------------------------------------#
        cur = 0
        self.block1 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[0]
                )
                for i in range(depths[0])
            ]
        )
        self.norm1 = norm_layer(embed_dims[0])

        # ----------------------------------#
        #   block2
        # ----------------------------------#
        # -----------------------------------------------#
        #   对输入图像进行分区，并下采样
        #   128, 128, 32 => 64, 64, 64 => 4096, 64
        # -----------------------------------------------#
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        # -----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   4096, 64 => 4096, 64
        # -----------------------------------------------#
        cur += depths[0]
        self.block2 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[1]
                )
                for i in range(depths[1])
            ]
        )
        self.norm2 = norm_layer(embed_dims[1])

        # ----------------------------------#
        #   block3
        # ----------------------------------#
        # -----------------------------------------------#
        #   对输入图像进行分区，并下采样
        #   64, 64, 64 => 32, 32, 160 => 1024, 160
        # -----------------------------------------------#
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        # -----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   1024, 160 => 1024, 160
        # -----------------------------------------------#
        cur += depths[1]
        self.block3 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[2]
                )
                for i in range(depths[2])
            ]
        )
        self.norm3 = norm_layer(embed_dims[2])

        # ----------------------------------#
        #   block4
        # ----------------------------------#
        # -----------------------------------------------#
        #   对输入图像进行分区，并下采样
        #   32, 32, 160 => 16, 16, 256 => 256, 256
        # -----------------------------------------------#
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        # -----------------------------------------------#
        #   利用transformer模块进行特征提取
        #   256, 256 => 256, 256
        # -----------------------------------------------#
        cur += depths[2]
        self.block4 = nn.ModuleList(
            [
                Block(
                    dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                    sr_ratio=sr_ratios[3]
                )
                for i in range(depths[3])
            ]
        )
        self.norm4 = norm_layer(embed_dims[3])

        self.apply(self._init_weights)

    # 初始化权重和偏置的辅助函数
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)  # 初始化 Linear 层的权重为截断正态分布，标准差为 0.02
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 将 Linear 层的偏置初始化为 0
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)  # 将 LayerNorm 层的偏置初始化为 0
            nn.init.constant_(m.weight, 1.0)  # 将 LayerNorm 层的权重初始化为 1.0
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels  # 计算 Conv2d 层的输出通道数
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))  # 初始化 Conv2d 层的权重为正态分布，标准差为 sqrt(2.0 / fan_out)
            if m.bias is not None:
                m.bias.data.zero_()  # 将 Conv2d 层的偏置初始化为 0

    # 前向传播函数
    def forward(self, x):
        B = x.shape[0]
        outs = []

        # Block1
        x, H, W = self.patch_embed1.forward(x)  # 将输入图像进行分区并下采样
        for i, blk in enumerate(self.block1):  # 使用 Block1 进行特征提取
            x = blk.forward(x, H, W)
        x = self.norm1(x)  # 进行 LayerNorm 归一化
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # ----------------------------------#
        #   block2
        # ----------------------------------#
        x, H, W = self.patch_embed2.forward(x)
        for i, blk in enumerate(self.block2):
            x = blk.forward(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # ----------------------------------#
        #   block3
        # ----------------------------------#
        x, H, W = self.patch_embed3.forward(x)
        for i, blk in enumerate(self.block3):
            x = blk.forward(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # ----------------------------------#
        #   block4
        # ----------------------------------#
        x, H, W = self.patch_embed4.forward(x)
        for i, blk in enumerate(self.block4):
            x = blk.forward(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class mit_b0(MixVisionTransformer):
    def __init__(self, pretrained=False):
        super(mit_b0, self).__init__(
            embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b0_backbone_weights.pth"), strict=False)


class mit_b1(MixVisionTransformer):
    def __init__(self, pretrained=False):
        super(mit_b1, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b1_backbone_weights.pth"), strict=False)


class mit_b2(MixVisionTransformer):
    def __init__(self, pretrained=False):
        super(mit_b2, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/Usegformer_b2_backbone_weights.pth"), strict=False)


class mit_b3(MixVisionTransformer):
    def __init__(self, pretrained=False):
        super(mit_b3, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b3_backbone_weights.pth"), strict=False)


class mit_b4(MixVisionTransformer):
    def __init__(self, pretrained=False):
        super(mit_b4, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b4_backbone_weights.pth"), strict=False)


class mit_b5(MixVisionTransformer):
    def __init__(self, pretrained=False):
        super(mit_b5, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_data/segformer_b5_backbone_weights.pth"), strict=False)

#=======================================================================================================================
#decoder_head
#=======================================================================================================================
class MLP(nn.Module):
    """
    简单MLP模块
    一个线性投影层用于将输入的特征进行线性变换
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)  # 定义了一个线性层，用于将输入的维度从input_dim转换为embed_dim

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # 将输入数据展平并转置，以便于送入线性层进行处理
        x = self.proj(x)  # 通过定义的线性层proj进行维度转换
        return x


class ConvModule(nn.Module):
    """
    通用的卷积模块，包括一个卷积层、一个批量归一化层和一个激活函数
    to let the module training more stable and accelerate convergence
    """

    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):  # c1输入通道数，c2输出通道数，k卷积核大小，s步长，p填充，g组数，act是否使用激活函数
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)  # 定义一个二维卷积层，不使用偏置项
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)  # 定义一个二维批量归一化层
        # 如果act为真，定义ReLU激活函数；如果act是一个nn.Module的实例，就直接使用它；否则使用恒等映射
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))  # 将输入数据x通过卷积层，然后通过批量归一化层，最后通过激活函数，并返回输出

    def fuseforward(self, x):
        return self.act(self.conv(x))  # 将输入数据x通过卷积层和激活函数，并返回输出


class SegFormerHead(nn.Module):
    """
    组合MLP和卷积层来构建语义分割头部
    将不同尺度的特征图通过MLP变换到相同的嵌入空间，然后通过上采样和卷积融合模块来结合这些特征，最后通过卷积层产生最终的分割预测
    """

    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        # 定义四个MLP层，对4个不同尺度的特征图进行线性变换
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        # 定义一个卷积模块，用于融合4个不同尺度的特征图
        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        # 定义一个卷积层，用于最终的分类预测
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

        # 定义一个2D丢弃层，用于正则化
        self.dropout = nn.Dropout2d(dropout_ratio)
        """
        随机地将输入中的一些元素设置为零。这有助于减少神经网络的复杂性，防止模型过拟合训练数据。
        Dropout 在训练时会随机地关闭一些神经元，而在推理（测试）阶段则不进行这种随机关闭，以保留更多的信息。
        """

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs  # 从输入中解构出4个不同尺度的特征图

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        # 对c4进行MLP变换并上采样
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # 对c3进行MLP变换并上采样
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # 对c2进行MLP变换并上采样
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        # 对c1进行MLP变换并上采样
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        # 使用卷积模块将4个尺度的特征图融合在一起
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)  # 对融合后的特征图应用丢弃层
        x = self.linear_pred(x)  # 通过卷积层进行最终的分类预测

        return x

class SegFormer(nn.Module):
    """
    定义了一个SegFormer模型
    集成了骨干网络（用于特征提取）和解码头（用于分割）。骨干网络的种类和大小可以通过参数phi来指定。整个模型首先使用
    骨干网络从输入图像中提取特征，然后使用解码头对特征进行分割，最后通过双线性插值将分割结果调整到原始图像的大小。
    """

    def __init__(self, num_classes=21, phi='b0', pretrained=False):
        super(SegFormer, self).__init__()
        # -----------------------------------------
        # 根据输入的phi参数选择对应的输入通道数量，phi用于指定模型的种类和规模
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]  # phi指模型的种类规模
        # -----------------------------------------
        # 根据输入的phi参数选择对应的骨干网络（backbone），用于特征提取
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        # -----------------------------------------
        # 根据输入的phi参数选择对应的特征图维度
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        # 定义解码头（decode head），用于语义分割
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        # ----------------------
        # 首先获取输入图像的高和宽
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(inputs)  # 将输入图像通过骨干网络backbone进行前向传播
        x = self.decode_head.forward(x)  # 再通过decode_head进行前向传播

        # --------------------------
        # 通过插值操作interpolate将分割结果x调整到输入图像的原始大小
        # mode=‘bilinear’ 即双线性插值的方法 平滑且保留细节
        # align_corners=True 即图片的四个角点对齐
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

if __name__=="__main__":
    model =  SegFormer(num_classes=21, phi='b2', pretrained=False)

    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)

    print("output shape", output.shape)