# -*- coding: UTF-8 -*-
"""
===================================================================================
@author : Leoda
@Date   : 2024/01/17 15:20:29
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

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
   return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

class ImprovedImportanceEvaluationNet(nn.Module):
    """
    Args:

    """
    def __init__(self, embed_dim= 64):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1)
        self.attention_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(embed_dim // 2, embed_dim)
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

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.attention_pool(x)
        x = torch.flatten(x, 1)
        importance_scores = torch.sigmoid(self.fc(x))
        return importance_scores

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(self.patch_size[0] // 2, self.patch_size[1] // 2))
        self.importance = ImprovedImportanceEvaluationNet(embed_dim)
        self.norm = nn.LayerNorm(embed_dim//2)
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

    def select_and_mask_windows(self, x, importance_scores):
        """
        基于重要性得分选择窗口并应用掩码。
        参数:
        - x: 原始特征图，形状为 [batch_size, num_patches, embed_dim]
        - importance_scores: 每个窗口的重要性得分，形状为 [batch_size, num_patches]
        - threshold: 用于选择窗口的阈值
        返回:
        - masked_x: 掩码后的特征图
        - selected_indices: 被选中的窗口索引
        """
        batch_size, channels, height, width = x.size()

        num_important = channels // 2
        new_channels = channels // 2
        _, idx = importance_scores.sort(descending=True)
        threshold_indices = idx[:, num_important]
        thresholds = importance_scores.gather(1, threshold_indices.unsqueeze(1)).squeeze(1)
        masked_x = torch.zeros(batch_size, new_channels, height, width, dtype=x.dtype, device=x.device)
        select_indices = []

        for i in range(batch_size):
            # 对每个batch计算selected_indices
            selected_indices = importance_scores[i] > thresholds[i]
            true_select = selected_indices.nonzero(as_tuple=False).squeeze(1)
            if true_select.shape[0] < new_channels:
                if true_select.shape[0] == 0:
                    true_select = torch.arange(new_channels)  # 若一个也没选中，随便选几个
                else:
                    # 重复选中的通道直到达到所需数量
                    repeats = (new_channels // true_select.shape[0]) + 1
                    true_select = true_select.repeat(repeats)[:new_channels]
            masked_x[i, :, :, :] = x[i, true_select[:new_channels], :, :]
            select_indices.append(true_select)

        return masked_x, select_indices

    def forward(self, x):
        x_global = self.proj(x)
        _, _, H, W = x_global.shape

        importance_scores = self.importance(x_global)
        x_local, select_indices = self.select_and_mask_windows(x_global, importance_scores)

        x_global = x_local
        x_local = x_local.flatten(2).transpose(1, 2)
        x_global = x_global.flatten(2).transpose(1, 2)
        x_local = self.norm(x_local)
        x_global = self.norm(x_global)
        return x_global, x_local, H, W

class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
                            nn.Conv2d(dim, dim, 1, 1, 0),
                            nn.SiLU(),
                            nn.Conv2d(dim, dim, 1, 1, 0)
                         )
    def forward(self, x):
        return self.act_block(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim//2, dim//2, bias=qkv_bias)
        self.kv = nn.Linear(dim//2, dim, bias=qkv_bias)

        self.to_qkv_local = nn.Conv2d(dim//2, dim//2 * 3, 1, bias=qkv_bias)
        self.attn_block_local = AttnMap(dim // 2)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim//2, dim//2, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim//2)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def global_attention(self, x_global, H, W):
        B, N, C = x_global.shape
        q = self.q(x_global).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # bs, num_heads, N, 32
        if self.sr_ratio > 1:
            # bs, 16384, 32 => bs, 32, 128, 128
            x_global = x_global.permute(0, 2, 1).reshape(B, C, H, W)
            # bs, 32, 128, 128 => bs, 32, 16, 16 => bs, 256, 32
            x_global = self.sr(x_global).reshape(B, C, -1).permute(0, 2, 1)
            x_global = self.norm(x_global)
            # bs, 256, 32 => bs, 256, 64 => bs, 256, 2, 1, 32 => 2, bs, 1, 256, 32
            kv = self.kv(x_global).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x_global).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x_global = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x_global

    def local_attention(self, x_local, H, W):
        B, N, C = x_local.shape
        x_local = self.q(x_local).permute(0, 2, 1).view(B, C, H, W)

        qkv = self.to_qkv_local(x_local)
        qkv = qkv.reshape(3, B, C, H, W)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = self.attn_block_local(q * k) * self.scale
        attn = self.attn_drop(torch.tanh(attn))
        x_local = (attn * v).reshape(B, -1, H, W).permute(0, 2, 3, 1).reshape(B, N, C)
        return x_local

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

    def forward(self, x_global, x_local, H, W):

        x_global = self.global_attention(x_global, H, W)
        x_local = self.local_attention(x_local, H, W)
        concat_out = torch.cat((x_global, x_local), dim=-1)

        x = self.proj(concat_out)
        x = self.proj_drop(x)

        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim//2)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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

    def forward(self, x_global, x_local, H, W):
        x_global = self.norm1(x_global)
        x_local = self.norm1(x_local)

        x = torch.cat((x_global, x_local), dim=-1)
        x = x + self.drop_path(self.attn(x_global, x_local, H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

# ----------------------------------#
#   Transformer模块，共有四个部分
# ----------------------------------#
class MixVisionTransformer(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # ----------------------------------#
        #   block1
        #   512, 512, 3 => 128, 128, 64 => 16384, 64
        # -----------------------------------------------#
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        # -----------------------------------------------#
        #   16384, 64 => 16384, 64
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
        #   128, 128, 64 => 64, 64, 128 => 4096, 128
        # -----------------------------------------------#
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        # -----------------------------------------------#
        #   4096, 128 => 4096, 128
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
        #   64, 64, 128 => 32, 32, 320 => 1024, 320
        # -----------------------------------------------#
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        # -----------------------------------------------#
        #   1024, 320 => 1024, 320
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
        #   32, 32, 320 => 16, 16, 512 => 256, 512
        # -----------------------------------------------#
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        # -----------------------------------------------#
        #   256, 512 => 256, 512
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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
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
        """
        Args:
            x: (B, 3, 512, 512)
        """
        B = x.shape[0]
        outs = []

        # Block1
        x_global, x_local, H, W = self.patch_embed1.forward(x)         #   512, 512, 3 => 128, 128, 32 => 16384, 32
        for i, blk in enumerate(self.block1):
            x = blk.forward(x_global, x_local, H, W)
        x = self.norm1(x) # bs, 16384, 32
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x) # bs, 32, 128, 128

        # ----------------------------------#
        #   block2
        # ----------------------------------#
        x_global, x_local, H, W = self.patch_embed2.forward(x)
        for i, blk in enumerate(self.block2):
            x = blk.forward(x_global, x_local, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # ----------------------------------#
        #   block3
        # ----------------------------------#
        x_global, x_local, H, W = self.patch_embed3.forward(x)
        for i, blk in enumerate(self.block3):
            x = blk.forward(x_global, x_local, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # ----------------------------------#
        #   block4
        # ----------------------------------#
        x_global, x_local, H, W = self.patch_embed4.forward(x)
        for i, blk in enumerate(self.block4):
            x = blk.forward(x_global, x_local, H, W)
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
            self.load_state_dict(torch.load("model_weights/segformer_b0_backbone_weights.pth"), strict=False)


class mit_b1(MixVisionTransformer):
    def __init__(self, pretrained=False):
        super(mit_b1, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_weights/segformer_b1_backbone_weights.pth"), strict=False)

class mit_b2(MixVisionTransformer):
    def __init__(self, pretrained=False):
        super(mit_b2, self).__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
        if pretrained:
            print("Load backbone weights")
            self.load_state_dict(torch.load("model_weights/UTFormer_b2_backbone_weights.pth"), strict=False)

#=======================================================================================================================
#decoder_head
#=======================================================================================================================
class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class ConvModule(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class UTFormerHead(nn.Module):
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(UTFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x

class UTFormer(nn.Module):
    def __init__(self, num_classes=21, phi='b2', pretrained=False):
        super(UTFormer, self).__init__()
        self.in_channels = {'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512]}[phi]
        self.backbone      = {'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2}[phi](pretrained)
        self.embedding_dim = {'b0': 256, 'b1': 256, 'b2': 768}[phi]
        self.decode_head = UTFormerHead(num_classes, self.in_channels, self.embedding_dim)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

if __name__=="__main__":
    model =  UTFormer(num_classes=21, phi='b2', pretrained=False)

    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)

    print("input shape", dummy_input)
    print("output shape", output.shape)