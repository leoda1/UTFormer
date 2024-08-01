# -*- coding: UTF-8 -*-
"""
===================================================================================
@author : Leoda
@Date   : 2024/04/16 11:02:39
@Project -> : learn$
==================================================================================
"""
import torch
from torch import nn
import warnings
import math
import torch.nn.functional as F

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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.to_qkv_local = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_block_local = AttnMap(dim // num_heads)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim * 2, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.apply(self._init_weights)

    def global_attention(self, x_global, H, W):
        B, N, C = x_global.shape
        # bs, 16384, 32 => bs, 16384, 32 => bs, 16384, 8, 4 => bs, 8, 16384, 4
        q = self.q(x_global).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 计算Q

        if self.sr_ratio > 1:
            # bs, 16384, 32 => bs, 32, 128, 128
            x_ = x_global.permute(0, 2, 1).reshape(B, C, H, W)
            # bs, 32, 128, 128 => bs, 32, 16, 16 => bs, 256, 32
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            # bs, 256, 32 => bs, 256, 64 => bs, 256,  => 2, bs, 8, 256, 4
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x_global).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x_global = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x_global

    def local_attention(self, x_local, H, W):
        B, N, C = x_local.shape
        x_local = x_local.permute(0, 2, 1).view(B, C, H, W)

        qkv = self.to_qkv_local(x_local)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H, W)
        q, k, v = qkv.unbind(1)

        attn = self.attn_block_local(q * k)
        attn = F.tanh(attn, dim=-1)
        x_local = attn * v
        x_local = x_local.reshape(B, -1, H, W).permute(0, 2, 3, 1).reshape(B, N, C)
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


if __name__ == '__main__':
    x = torch.randn(1, 64, 128, 128)
    net = ()
    x = net(x)