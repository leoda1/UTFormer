# -*- coding: UTF-8 -*-
"""
===================================================================================
@author : Leoda
@Date   : 2024/01/23 21:12:43
@Project -> : learn$
==================================================================================
"""
import math
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_


def knn_keops(query, database, k, return_dist=False):
    """
    Compute k-nearest neighborsusing the Keops library
    Backward pass turned off; Keops does not provide backward pass for distance
    Args:
        query - b x n_ x c, the position of tokens looking for knn
        database - b x n x c, the candidate tokens for knn
        k - int, the nunmber of neighbors to be found
        return_dist - bool, whether to return distance to the neighbors
    Returns:
        nn_dix - b x n x k, the indices of the knn
        nn_dist - b x n x k, if return_dist, the distance to the knn
    """
    b, n, c = database.shape
    with torch.no_grad():
        query = query.detach()
        database = database.detach()
        # Keops does not support half precision
        if query.dtype != torch.float32:
            query = query.to(torch.float32)
        if database.dtype != torch.float32:
            database = database.to(torch.float32)
        from pykeops.torch import LazyTensor
        query_ = LazyTensor(query[:, None, :, :])
        database_ = LazyTensor(database[:, :, None, :])
        dist = ((query_-database_) ** 2).sum(-1) ** 0.5  # b x n x n_
    if return_dist:
        nn_dist, nn_idx = dist.Kmin_argKmin(k, dim=1)  # b x n_ x k
        return nn_idx, nn_dist
    else:
        nn_idx = dist.argKmin(k, dim=1)  # b x n_ x k
        return nn_idx

def space_filling_cluster(pos, m, h, w, no_reorder=False, sf_type='', use_anchor=True):
    """
    The balanced clustering algorithm based on space-filling curves
    In the case where number of tokens not divisible by cluster size,
    the last cluster will have a few blank spots, indicated by the mask returned
    Args:
        pos - b x n x 2, 表示n个token的位置
        m - int, 目标聚类大小
        h,w - int, 图像高宽
        no_reorder - bool, if True, return the clustering based on the original order of tokens;
                            otherwise, reorder the tokens so that the same cluster stays together是否重排token
        sf_type - str, can be 'peano' or 'hilbert', or otherwise, horizontal scanlines w/ alternating
                        direction in each row by default空间填充曲线类型,'peano'或'hilbert'
        use_anchor - bool, whether to use space-fiiling anchors or not; if False, directly compute
                            space-filling curves on the token positions 是否使用anchor
    Returns:
        pos - b x n x 2, returned only if no_reorder is False; the reordered position of tokens重排后的token位置
        cluster_mean_pos - b x k x 2, the clustering centers聚类中心
        member_idx - b x k x m, the indices of tokens in each cluster 每个聚类中的token索引
        cluster_mask - b x k x m, the binary mask indicating the paddings in last cluster (0 if padding) 最后聚类的填充mask
        pos_ranking - b x n x 1, returned only if no_reorder is False; i-th entry is the idx of the token
                                rank i in the new order 重排后token的排名
    """
    with torch.no_grad():
        pos = pos.detach()
        if pos.dtype != torch.float:
            pos = pos.to(torch.float)
        b, n, d = pos.shape #b x n x 2

        k = int(math.ceil(n / m))

        if use_anchor:
            patch_len = (h * w / k) ** 0.5

            num_patch_h = int(round(h / patch_len))
            num_patch_w = int(round(w / patch_len))

            patch_len_h, patch_len_w = h / num_patch_h, w / num_patch_w

            if sf_type == 'peano':
                num_patch_h = max(3, int(3 ** round(math.log(num_patch_h, 3))))
                patch_len_h = h / num_patch_h
                num_patch_w = int(round(w / h * 3) * (num_patch_h / 3))
                patch_len_w = w / num_patch_w
            elif sf_type == 'hilbert':
                num_patch_h = max(2, int(2 ** round(math.log(num_patch_h, 2))))
                patch_len_h = h / num_patch_h
                num_patch_w = int(round(w / h * 2) * (num_patch_h / 2))
                patch_len_w = w / num_patch_w

            hs = torch.arange(0, num_patch_h, device=pos.device)
            ws = torch.arange(0, num_patch_w, device=pos.device)

            ys, xs = torch.meshgrid(hs, ws)

            grid_pos = torch.stack([xs, ys], dim=2)

            grid_pos = grid_pos.reshape(-1, 2)

            if  sf_type == 'hilbert':
                order_grid_idx, order_idx = calculate_hilbert_order(num_patch_h, num_patch_w, grid_pos.unsqueeze(0))
                order_grid_idx = order_grid_idx[0]
                order_idx = order_idx[0]
            else:
                order_mask = torch.ones_like(ys)
                order_mask[1::2] = -1

                order_mask = order_mask * xs
                order_mask = order_mask + ys * w

                order_mask[1::2] += (w - 1)

                order_mask = order_mask.reshape(-1)

                order_idx = order_mask.sort()[1]

                order_idx_src = torch.arange(len(order_idx)).to(pos.device)

                order_grid_idx = torch.zeros_like(order_idx_src)

                order_grid_idx.scatter_(index=order_idx, dim=0, src=order_idx_src)

            ordered_grid = grid_pos[order_idx]
            patch_len_hw = torch.Tensor([patch_len_w, patch_len_h]).to(pos.device)
            init_pos_means = ordered_grid * patch_len_hw + patch_len_hw / 2 - 0.5
            nump = ordered_grid.shape[0]
            prev_means = torch.zeros_like(init_pos_means)
            prev_means[1:] = init_pos_means[:nump - 1].clone()
            prev_means[0] = prev_means[1] - (prev_means[2] - prev_means[1])
            next_means = torch.zeros_like(init_pos_means)
            next_means[:nump - 1] = init_pos_means[1:].clone()
            next_means[-1] = next_means[-2] + (next_means[-2] - next_means[-3])
            mean_assignment = (pos / patch_len_hw).floor()
            mean_assignment = mean_assignment[..., 0] + mean_assignment[..., 1] * num_patch_w
            mean_assignment = order_grid_idx.unsqueeze(0).expand(b, -1).gather(index=mean_assignment.long(),
                                                                               dim=1).unsqueeze(2)
            prev_mean_assign = prev_means.unsqueeze(0).expand(b, -1, -1).gather(index=mean_assignment.expand(-1, -1, d),
                                                                                dim=1)
            next_mean_assign = next_means.unsqueeze(0).expand(b, -1, -1).gather(index=mean_assignment.expand(-1, -1, d),
                                                                                dim=1)
            dist_prev = (pos - prev_mean_assign).pow(2).sum(-1)
            dist_next = (pos - next_mean_assign).pow(2).sum(-1)
            dist_ratio = dist_prev / (dist_next + 1e-5)
            pos_ranking = mean_assignment * (dist_ratio.max() + 1) + dist_ratio.unsqueeze(2)

            pos_ranking = pos_ranking.sort(dim=1)[1]


            if sf_type == 'hilbert':
                _, pos_ranking = calculate_hilbert_order(h, w, pos)
            else:
                hs = torch.arange(0, h, device=pos.device)
                ws = torch.arange(0, w, device=pos.device)
                ys, xs = torch.meshgrid(hs, ws)

                order_mask = torch.ones_like(ys)  # h x w
                order_mask[1::2] = -1
                order_mask = order_mask * xs
                order_mask = order_mask + ys * w
                order_mask[1::2] += (w - 1)
                order_mask = order_mask.reshape(-1)

                pos_idx = pos[..., 0] + pos[..., 1] * w
                order_mask = order_mask.gather(index=pos_idx.long().reshape(-1), dim=0).reshape(b, n)

                pos_ranking = order_mask.sort()[1]

            pos_ranking = pos_ranking.unsqueeze(2)

        pos = pos.gather(index=pos_ranking.expand(-1, -1, d), dim=1)

        if k * m == n:
            cluster_mask = None
            cluster_mean_pos = pos.reshape(b, k, -1, d).mean(2)
        else:
            pos_pad = torch.zeros(b, k * m, d, dtype=pos.dtype, device=pos.device)
            pos_pad[:, :n] = pos.clone()

            cluster_mask = torch.zeros(b, k * m, device=pos.device).long()
            cluster_mask[:, :n] = 1

            cluster_mask = cluster_mask.reshape(b, k, m)

            cluster_mean_pos = pos_pad.reshape(b, k, -1, d).sum(2) / cluster_mask.sum(2, keepdim=True)

        if no_reorder:
            if k * m == n:
                member_idx = pos_ranking.reshape(b, k, m)
            else:
                member_idx = torch.zeros(b, k * m, device=pos.device, dtype=torch.int64)
                member_idx[:, :n] = pos_ranking.squeeze(2)
                member_idx = member_idx.reshape(b, k, m)
            return cluster_mean_pos, member_idx, cluster_mask
        else:
            member_idx = torch.arange(k * m, device=pos.device)
            member_idx[n:] = 0
            member_idx = member_idx.unsqueeze(0).expand(b, -1)
            member_idx = member_idx.reshape(b, k, m)
            return pos, cluster_mean_pos, member_idx, cluster_mask, pos_ranking

def calculate_hilbert_order(h, w, pos):
    """
    Given height and width of the canvas and position of tokens,
    calculate the hilber curve order of the tokens
    Args:
        h,w - int, height and width
        pos - b x n x 2, positions of tokens
    Returns:
        final_order_ - b x n, i-th entry is the rank of i-th token in the new order
        final_order_index - b x n, i-th entry is the idx of the token rank i in the new order
    """
    b, n, _ = pos.shape
    # 获取pos的维度：b是批次大小，n是点的数量

    num_levels = math.ceil(math.log(h, 2))
    # 计算希尔伯特曲线的层数，基于高度h

    assert num_levels >= 1, "h too short"
    # 确保层数至少为1

    first_w = None
    # 初始化变量first_w，用于处理非正方形画布的情况

    if h != w:
        first_w = round(2 * (w / h))
        # 如果高度和宽度不同，计算宽度与高度的比例

        if first_w == 2:
            first_w = None
    # 如果宽度是高度的两倍，则将first_w设置为None

    rotate_dict = torch.Tensor([[[-1, 1], [0, 0]], [[0, -1], [0, 1]], [[1, 0], [-1, 0]]]).to(pos.device)
    # 创建一个旋转字典，用于确定如何在每一层旋转点

    if first_w is not None:
        rotate_dict_f = rotate_dict[0].repeat(1, math.ceil(first_w / 2))[:, :first_w]
        # 如果first_w不为None，则创建一个特殊的旋转字典rotate_dict_f

        rotate_dict_f = rotate_dict_f.reshape(-1)
        # 将rotate_dict_f重塑为一维数组

    rotate_dict = rotate_dict.reshape(3, -1)
    # 将rotate_dict重塑为一维数组

    rot_res_dict = torch.Tensor([[0, 3, 1, 2], [2, 3, 1, 0], [2, 1, 3, 0], [0, 1, 3, 2]]).to(pos.device)
    # 创建一个结果旋转字典rot_res_dict

    last_h = h
    # 初始化last_h为画布高度

    rem_pos = pos
    # 初始化rem_pos为点的位置

    levels_pos = []
    # 初始化存储每层位置的列表

    for le in range(num_levels):
        # 对每个层级进行遍历

        cur_h = last_h / 2
        # 当前层的高度为上一层的一半

        level_pos = (rem_pos / cur_h).floor()
        # 计算当前层的位置

        levels_pos.append(level_pos)
        # 将当前层的位置添加到列表中

        rem_pos = rem_pos % cur_h
        # 更新剩余的位置

        last_h = cur_h
        # 更新last_h为当前层的高度

    orders = []
    # 初始化存储每层排序的列表

    for i in range(len(levels_pos)):
        # 对每个层级的位置进行遍历

        level_pos = levels_pos[i]
        # 获取当前层的位置

        if i == 0 and first_w is not None:
            level_pos_index = level_pos[..., 0] + level_pos[..., 1] * first_w
            # 如果是第一层且first_w不为None，计算位置索引

        else:
            level_pos_index = level_pos[..., 0] + level_pos[..., 1] * 2
            # 否则，计算位置索引

        rotate = torch.zeros_like(pos[..., 0])
        # 初始化旋转矩阵

        for j in range(i):
            # 对当前层之前的每个层级进行遍历

            cur_level_pos = levels_pos[j]
            # 获取当前层的位置

            if j == 0 and first_w is not None:
                cur_level_pos_index = cur_level_pos[..., 0] + cur_level_pos[..., 1] * first_w  # b x n
                # 如果是第一层且first_w不为None，计算当前层级的位置索引

                cur_rotate = rotate_dict_f.gather(index=cur_level_pos_index.long().view(-1), dim=0).reshape(b, n)
                # 根据位置索引收集旋转方向
            else:
                rotate_d = rotate_dict.gather(index=(rotate % 3).long().view(-1, 1).expand(-1, 4), dim=0).reshape(b, n,
                                                                                                                  4)
                # 收集当前旋转状态对应的旋转方向

                cur_level_pos_index = cur_level_pos[..., 0] + cur_level_pos[..., 1] * 2  # b x n
                # 计算当前层级的位置索引

                cur_rotate = rotate_d.gather(index=cur_level_pos_index.long().unsqueeze(2), dim=2).reshape(b, n)
                # 根据位置索引收集旋转方向

            rotate = cur_rotate + rotate
            # 更新旋转矩阵

            rotate = rotate % 4
            # 确保旋转矩阵的值在0到3之间

            rotate_res = rot_res_dict.gather(index=rotate.long().view(-1, 1).expand(-1, 4), dim=0).reshape(b, n, 4)
            # 根据当前旋转矩阵获取旋转结果

            rotate_res = rotate_res.gather(index=level_pos_index.long().unsqueeze(2), dim=2).squeeze(2)  # b x n
            # 根据当前层的位置索引收集旋转结果

            orders.append(rotate_res)
            # 将旋转结果添加到orders列表

        final_order = orders[-1]
        # 获取最后一层的旋转结果作为最终排序

        for i in range(len(orders) - 1):
            cur_order = orders[i]
            final_order = final_order + cur_order * (4 ** (num_levels - i - 1))
            # 根据每层的旋转结果和层数计算最终的排序

        final_order_index = final_order.sort(dim=1)[1]
        # 对最终排序进行排序，获取排序索引

        order_src = torch.arange(n).to(pos.device).unsqueeze(0).expand(b, -1)  # b x n
        # 创建原始的排序索引

        final_order_ = torch.zeros_like(order_src)
        # 初始化最终排序的张量

        final_order_.scatter_(index=final_order_index, src=order_src, dim=1)
        # 将原始排序索引映射到最终排序的位置

        return final_order_, final_order_index
        # 返回最终排序和排序索引

class ClusterTransformerBlock(nn.Module):
    r""" Cluster Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        layer_scale (float, optional): Layer scale initial parameter. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads,
                 mlp_ratio=2., drop=0., attn_drop=0., drop_path=0., layer_scale=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = ClusterAttention(
            dim, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # layer_scale code copied from https://github.com/SHI-Labs/Neighborhood-Attention-Transformer/blob/a2cfef599fffd36d058a5a4cfdbd81c008e1c349/classification/nat.py
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float] and layer_scale > 0:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

    def forward(self, feat, member_idx, cluster_mask, pe_idx, global_attn):
        """
        Args:
            feat - b x n x c, token features
            member_idx - b x n x nbhd, token idx in each local nbhd
            cluster_mask - b x n x nbhd, binary mask for valid tokens (1 if valid)
            pe_idx - b x n x nbhd, idx for the pre-computed position embedding lookup table
            global_attn - bool, whether to perform global attention
        """

        b, n, c = feat.shape
        assert c == self.dim, "dim does not accord to input"

        shortcut = feat
        feat = self.norm1(feat)

        # cluster attention
        feat = self.attn(feat=feat,
                         member_idx=member_idx,
                         cluster_mask=cluster_mask,
                         pe_idx=pe_idx,
                         global_attn=global_attn)

        # FFN
        if not self.layer_scale:
            feat = shortcut + self.drop_path(feat)
            feat_mlp = self.mlp(self.norm2(feat))
            feat = feat + self.drop_path(feat_mlp)
        else:
            feat = shortcut + self.drop_path(self.gamma1 * feat)
            feat_mlp = self.mlp(self.norm2(feat))
            feat = feat + self.drop_path(self.gamma2 * feat_mlp)

        return feat

class ClusterAttention(nn.Module):
    """
    Performs local attention on nearest clusters

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.pos_dim = 2
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2*dim)
        self.softmax = nn.Softmax(dim=-1)

        self.blank_k = nn.Parameter(torch.randn(dim))
        self.blank_v = nn.Parameter(torch.randn(dim))

        self.pos_embed = nn.Linear(self.pos_dim+3, num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, feat, member_idx, cluster_mask, pe_idx, global_attn):
        """
        Args:
            feat - b x n x c, token features
            member_idx - b x n x nbhd, token idx in each local nbhd
            cluster_mask - b x n x nbhd, binary mask for valid tokens (1 if valid)
            pe_idx - b x n x nbhd, idx for the pre-computed position embedding lookup table
            global_attn - bool, whether to perform global attention
        """

        b, n, c = feat.shape
        c_ = c // self.num_heads
        assert c == self.dim, "dim does not accord to input"
        h = self.num_heads

        # get qkv
        q = self.q(feat)  # b x n x c
        q = q * self.scale
        kv = self.kv(feat)  # b x n x 2c

        # get attention
        if global_attn:
            q = q.reshape(b, n, h, -1).permute(0, 2, 1, 3)  # b x h x n x c_
            kv = kv.view(b, n, h, 2, c_).permute(3, 0, 2, 1, 4)  # 2 x b x h x n x c_
            key, v = kv[0], kv[1]
            attn = q @ key.transpose(-1, -2)  # b x h x n x n
            mask = None
        else:
            nbhd_size = member_idx.shape[-1]
            m = nbhd_size
            q = q.reshape(b, n, h, -1).permute(0, 2, 1, 3)
            kv = kv.view(b, n, h, 2, c_).permute(3, 0, 2, 1, 4)  # 2 x b x h x n x c_
            key, v = kv[0], kv[1]
            attn = CLUSTENQKFunction.apply(q, key, member_idx)  # b x h x n x m
            mask = cluster_mask
            if mask is not None:
                mask = mask.reshape(b, 1, n, m)

        # position embedding
        global pre_table
        if not pre_table.is_cuda:
            pre_table = pre_table.to(pe_idx.device)
        pe_table = self.pos_embed(pre_table)  # 111 x 111 x h for img_size 224x224

        pe_shape = pe_idx.shape
        pos_embed = pe_table.gather(index=pe_idx.view(-1, 1).expand(-1, h), dim=0).reshape(*(pe_shape), h).permute(0, 3, 1, 2)

        attn = attn + pos_embed

        if mask is not None:
            attn = attn + (1-mask)*(-100)

        # blank token
        blank_attn = (q * self.blank_k.reshape(1, h, 1, c_)).sum(-1, keepdim=True)  # b x h x n x 1
        attn = torch.cat([attn, blank_attn], dim=-1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        blank_attn = attn[..., -1:]
        attn = attn[..., :-1]
        blank_v = blank_attn * self.blank_v.reshape(1, h, 1, c_)  # b x h x n x c_

        # aggregate v
        if global_attn:
            feat = (attn @ v).permute(0, 2, 1, 3).reshape(b, n, c)
            feat = feat + blank_v.permute(0, 2, 1, 3).reshape(b, n, c)
        else:
            feat = CLUSTENAVFunction.apply(attn, v, member_idx).permute(0, 2, 1, 3).reshape(b, n, c)
            feat = feat + blank_v.permute(0, 2, 1, 3).reshape(b, n, c)

        feat = self.proj(feat)
        feat = self.proj_drop(feat)

        return feat

class BasicLayer(nn.Module):
    """ AutoFocusFormer layer for one stage.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        cluster_size (int): Cluster size.
        nbhd_size (int): Neighbor size. If larger than or equal to number of tokens, perform global attention;
                            otherwise, rounded to the nearest multiples of cluster_size.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        alpha (float, optional): the weight to be multiplied with importance scores. Default: 4.0
        ds_rate (float, optional): downsampling rate, to be multiplied with the number of tokens. Default: 0.25
        reserve_on (bool, optional): whether to turn on reserve tokens in downsampling. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        layer_scale (float, optional): Layer scale initial parameter. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self, dim, out_dim, cluster_size, nbhd_size,
                 depth, num_heads, mlp_ratio,
                 alpha=4.0, ds_rate=0.25, reserve_on=True,
                 drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 layer_scale=0.0, downsample=None):

        super().__init__()

        self.dim = dim
        self.nbhd_size = nbhd_size
        self.cluster_size = cluster_size
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            ClusterTransformerBlock(dim=dim,
                                    num_heads=num_heads,
                                    mlp_ratio=mlp_ratio,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    layer_scale=layer_scale,
                                    norm_layer=norm_layer)
            for i in range(depth)])

        # merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, out_dim=out_dim, norm_layer=norm_layer, alpha=alpha, ds_rate=ds_rate, reserve_on=reserve_on)
        else:
            self.downsample = None

        # cache the clustering result for the first feature map since it is on grid
        self.pos, self.cluster_mean_pos, self.member_idx, self.cluster_mask, self.reorder = None, None, None, None, None
        # fc for importance scores
        if downsample is not None:
            self.prob_net = nn.Linear(dim, 1)

    def forward(self, pos, feat, h, w, on_grid, stride):
        """
        Args:
            pos - 位置矩阵，形状为 (b, n, 2)
            feat - 特征矩阵，形状为 (b, n, c)
            h,w - token位置的最大高度和宽度
            on_grid - bool, whether the tokens are still on grid; True for the first feature map
            stride - int, "stride" of the current token set; starts with 2, then doubles in each stage
        """
        b, n, d = pos.shape
        c = feat.shape[2]

        assert self.cluster_size > 0, 'self.cluster_size must be positive'

        if self.nbhd_size >= n:
            global_attn = True
            member_idx, cluster_mask = None, None
        else:
            global_attn = False
            k = int(math.ceil(n / float(self.cluster_size)))  # number of clusters
            nnc = min(int(round(self.nbhd_size / float(self.cluster_size))), k)  # number of nearest clusters
            nbhd_size = self.cluster_size * nnc

        if global_attn:
            rel_pos = (pos[:, None, :, :] + rel_pos_width) - pos[:, :, None, :]  # b x n x n x d
        else:
            if k == n:
                cluster_mean_pos = pos
                member_idx = torch.arange(n, device=feat.device).long().reshape(1, n, 1).expand(b, -1, -1)
                cluster_mask = None
            else:
                if on_grid:
                    if self.cluster_mean_pos is None:
                        self.pos, self.cluster_mean_pos, self.member_idx, self.cluster_mask, self.reorder = space_filling_cluster(
                            pos, self.cluster_size, h, w, no_reorder=False)
                    pos, cluster_mean_pos, member_idx, cluster_mask = self.pos[:b], self.cluster_mean_pos[
                                                                                    :b], self.member_idx[
                                                                                         :b], self.cluster_mask
                    feat = feat[
                        torch.arange(b).to(feat.device).repeat_interleave(n), self.reorder[:b].view(-1)].reshape(b, n,
                                                                                                                 c)
                    if cluster_mask is not None:
                        cluster_mask = cluster_mask[:b]
                else:
                    pos, cluster_mean_pos, member_idx, cluster_mask, reorder = space_filling_cluster(pos,
                                                                                                     self.cluster_size,
                                                                                                     h, w,
                                                                                                     no_reorder=False)
                    feat = feat[torch.arange(b).to(feat.device).repeat_interleave(n), reorder.view(-1)].reshape(b, n, c)

            assert member_idx.shape[1] == k and member_idx.shape[2] == self.cluster_size, "member_idx shape incorrect!"
            nearest_cluster = knn_keops(pos, cluster_mean_pos, nnc)  # b x n x nnc

            # collect neighbor indices from nearest clusters
            m = self.cluster_size
            member_idx = member_idx.gather(index=nearest_cluster.view(b, -1, 1).expand(-1, -1, m), dim=1).reshape(b, n,
                                                                                                                  nbhd_size)
            if cluster_mask is not None:
                cluster_mask = cluster_mask.gather(index=nearest_cluster.view(b, -1, 1).expand(-1, -1, m),
                                                   dim=1).reshape(b, n, nbhd_size)
            pos_ = pos.gather(index=member_idx.view(b, -1, 1).expand(-1, -1, d), dim=1).reshape(b, n, nbhd_size, d)
            rel_pos = pos_ - (pos.unsqueeze(2) - rel_pos_width)  # b x n x nbhd_size x d

        # compute indices in the position embedding lookup table
        pe_idx = (rel_pos[..., 1] * table_width + rel_pos[..., 0]).long()

        for i_blk in range(len(self.blocks)):
            blk = self.blocks[i_blk]
            feat = blk(feat=feat,
                       member_idx=member_idx,
                       cluster_mask=cluster_mask,
                       pe_idx=pe_idx,
                       global_attn=global_attn)

        if self.downsample is not None:
            learned_prob = self.prob_net(feat).sigmoid()  # b x n x 1
            reserve_num = math.ceil(h / (stride * 2)) * math.ceil(w / (stride * 2))

            pos, feat = self.downsample(pos=pos, feat=feat,
                                        member_idx=member_idx, cluster_mask=cluster_mask,
                                        learned_prob=learned_prob, stride=stride,
                                        pe_idx=pe_idx, reserve_num=reserve_num)
        return pos, feat

class ClusterMerging(nn.Module):
    r""" Adaptive Downsampling.

    Args:
        dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        alpha (float, optional): the weight to be multiplied with importance scores. Default: 4.0
        ds_rate (float, optional): downsampling rate, to be multiplied with the number of tokens. Default: 0.25
        reserve_on (bool, optional): whether to turn on reserve tokens in downsampling. Default: True
    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm, alpha=4.0, ds_rate=0.25, reserve_on=True):
        super().__init__()
        self.dim = dim
        self.pos_dim = 2
        self.alpha = alpha
        self.ds_rate = ds_rate
        self.reserve_on = reserve_on

        # pointconv
        inner_ch = 4
        self.weight_net = nn.Sequential(
            nn.Linear(self.pos_dim+3, inner_ch, bias=True),
            nn.LayerNorm(inner_ch),
            nn.GELU()
        )
        self.norm = norm_layer(inner_ch*dim)
        self.linear = nn.Linear(dim*inner_ch, out_dim)

    def forward(self, pos, feat, member_idx, cluster_mask, learned_prob, stride, pe_idx, reserve_num):
        """
        Args:
            pos - b x n x 2, token positions
            feat - b x n x c, token features
            member_idx - b x n x nbhd, token idx in each local nbhd
            cluster_mask - b x n x nbhd, binary mask for valid tokens (1 if valid)
            learned_prob - b x n x 1, learned importance scores
            stride - int, "stride" of the current feature map, 2,4,8 for the 3 stages respectively
            pe_idx - b x n x nbhd, idx for the pre-computed position embedding lookup table
            reserve_num - int, number of tokens to be reserved
        """

        b, n, c = feat.shape
        d = pos.shape[2]

        keep_num = int(n*self.ds_rate)

        # grid prior
        if stride == 2:  # no ada ds yet, no need ada grid
            grid_prob = ((pos % stride).sum(-1) == 0).float()  # b x n
        else:
            _, min_dist = knn_keops(pos, pos, 2, return_dist=True)  # b x n x 2
            min_dist = min_dist[:, :, 1]  # b x n
            ada_stride = 2**(min_dist.log2().ceil()+1)  # b x n
            grid_prob = ((pos.long() % ada_stride.unsqueeze(2).long()).sum(-1) == 0).float()  # b x n

        final_prob = grid_prob

        # add importance score
        if learned_prob is not None:
            lp = learned_prob.detach().view(b, n)
            lp = lp * self.alpha
            final_prob = final_prob + lp

        # reserve points on a coarse grid
        if self.reserve_on:
            reserve_mask = ((pos % (stride*2)).sum(-1) == 0).float()  # b x n
            final_prob = final_prob + (reserve_mask*(-100))
            sample_num = keep_num - reserve_num
        else:
            sample_num = keep_num

        # select topk tokens as merging centers
        sample_idx = final_prob.topk(sample_num, dim=1, sorted=False)[1]  # b x n_

        if self.reserve_on:
            reserve_idx = reserve_mask.nonzero(as_tuple=True)[1].reshape(b, reserve_num)
            idx = torch.cat([sample_idx, reserve_idx], dim=-1).unsqueeze(2)  # b x n_ x 1
        else:
            idx = sample_idx.unsqueeze(2)

        n = idx.shape[1]
        assert n == keep_num, "n not equal to keep num!"

        # gather pos, nbhd, nbhd position embedding, nbhd importance scores for topk merging locations
        pos = pos.gather(index=idx.expand(-1, -1, d), dim=1)  # b x n' x d

        nbhd_size = member_idx.shape[-1]
        member_idx = member_idx.gather(index=idx.expand(-1, -1, nbhd_size), dim=1)  # b x n' x m
        pe_idx = pe_idx.gather(index=idx.expand(-1, -1, nbhd_size), dim=1)  # b x n' x m
        if cluster_mask is not None:
            cluster_mask = cluster_mask.gather(index=idx.expand(-1, -1, nbhd_size), dim=1)  # b x n' x m
        if learned_prob is not None:
            lp = learned_prob.gather(index=member_idx.view(b, -1, 1), dim=1).reshape(b, n, nbhd_size, 1)  # b x n x m x 1

        # pointconv weights
        global pre_table
        if not pre_table.is_cuda:
            pre_table = pre_table.to(pe_idx.device)
        weights_table = self.weight_net(pre_table)  # 111 x 111 x ic

        weight_shape = pe_idx.shape
        inner_ch = weights_table.shape[-1]
        weights = weights_table.gather(index=pe_idx.view(-1, 1).expand(-1, inner_ch), dim=0).reshape(*(weight_shape), inner_ch)

        if learned_prob is not None:
            if cluster_mask is not None:
                lp = lp * cluster_mask.unsqueeze(3)
            weights = weights * lp
        else:
            if cluster_mask is not None:
                weights = weights * cluster_mask.unsqueeze(3)

        # merge features
        feat = CLUSTENWFFunction.apply(weights, feat, member_idx.view(b, n, -1)).reshape(b, n, -1)  # b x n x ic*c
        feat = self.norm(feat)
        feat = self.linear(feat)  # b x n x 2c

        return pos, feat

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of channels. Default: 32.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=32, norm_layer=None):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj1 = nn.Conv2d(in_chans, embed_dim//2, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(embed_dim//2)
        self.act1 = nn.GELU()
        self.proj2 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2, padding=1)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        Args:
            x - b x c x h x w, input imgs
        """

        x = self.proj2(self.act1(self.bn(self.proj1(x))))
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # b x n x c
        if self.norm is not None:
            x = self.norm(x)

        hs = torch.arange(0, h, device=x.device)
        ws = torch.arange(0, w, device=x.device)
        ys, xs = torch.meshgrid(hs, ws)
        pos = torch.stack([xs, ys], dim=2).unsqueeze(0).expand(b, -1, -1, -1).reshape(b, -1, 2).to(x.dtype)

        return pos, x, h, w

class AutoFocusFormer(nn.Module):
    """

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (tuple(int)): Feature dimension of each stage. Default: [32,128,256,512]
        cluster_size (int): Cluster size. Default: 8
        nbhd_size (tuple(int)): Neighborhood size of local attention of each stage. Default: [48,48,48,49]
        alpha (float, optional): the weight to be multiplied with importance scores. Default: 4.0
        ds_rate (float, optional): downsampling rate, to be multiplied with the number of tokens. Default: 0.25
        reserve_on (bool, optional): whether to turn on reserve tokens in downsampling. Default: True
        depths (tuple(int)): Depth of each AFF layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.0
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        layer_scale (float, optional): Layer scale initial parameter; turned off if 0.0. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
    """

    def __init__(self, in_chans=3, num_classes=1000, embed_dim=[32, 128, 256, 512],
                 cluster_size=8, nbhd_size=[48, 48, 48, 49],
                 alpha=4.0, ds_rate=0.25, reserve_on=True,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=2., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 layer_scale=0.0,
                 downsample=ClusterMerging,
                 img_size=224,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            in_chans=in_chans, embed_dim=embed_dim[0],
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        build_pe_lookup(img_size)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim[i_layer]),
                               out_dim=int(embed_dim[i_layer+1]) if (i_layer < self.num_layers - 1) else None,
                               cluster_size=cluster_size,
                               nbhd_size=nbhd_size[i_layer],
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               alpha=alpha,
                               ds_rate=ds_rate,
                               reserve_on=reserve_on,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=downsample if (i_layer < self.num_layers - 1) else None,
                               layer_scale=layer_scale)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {}

    def forward_features(self, x):
        '''
        x - b x c x h x w
        '''
        pos, x, h, w = self.patch_embed(x)  # b x n x c, b x n x d
        x = self.pos_drop(x)

        for i_layer in range(len(self.layers)):
            layer = self.layers[i_layer]
            pos, x = layer(pos, x, h=h, w=w, on_grid=i_layer == 0, stride=2**(i_layer+1))

        x = self.norm(x)  # b x n x c
        x = x.mean(1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

