# -*- coding: UTF-8 -*-
"""
===================================================================================
@author : Leoda
@Date   : 2024/01/24 15:40:53
@Project -> : learn$
==================================================================================
"""
import torch
from torch import nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import random
import math
import warnings


def set_random_seed(seed_value=42):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        self.proj2 = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1, stride=1)
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
        x_global2 = self.proj2(x_global)
        importance_scores = self.importance(x_global)
        x_local, select_indices = self.select_and_mask_windows(x_global, importance_scores)
        #
        # x_global = x_local
        # x_local = x_local.flatten(2).transpose(1, 2)
        # x_global = x_global.flatten(2).transpose(1, 2)
        # x_global2 = x_global2.flatten(2).transpose(1, 2)
        # x_local = self.norm(x_local)
        # x_global = self.norm(x_global)
        # x_global2 = self.norm(x_global2)
        return x_global2, x_local, H, W

def load_and_preprocess_image(image_path, image_size=512):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 调整图像大小
        transforms.ToTensor(),  # 将图像转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准化，使用ImageNet的均值和标准差
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式
    image = transform(image)
    return image.unsqueeze(0)


def test_model(patch_size=7, stride=4, in_chans=3, embed_dim=64, num_channels=32):
    image_path = "E:\\project\\learn\\img\\top_mosaic_09cm_area3_89.jpg"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OverlapPatchEmbed(patch_size, stride, in_chans, embed_dim)
    model.to(device)
    image = load_and_preprocess_image(image_path)
    image = image.to(device)

    global_features, local_features, H, W = model(image)
    # Prepare to plot the images
    num_rows = 4  # Total number of rows (4 rows for global features + 4 rows for local features)
    num_cols = 16  # Total number of columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(32, 8), dpi=300, gridspec_kw={'wspace':0.01, 'hspace':0.01})

    # Iterate over the first 32 channels
    for i in range(num_channels):
        row_global = i // num_cols
        col_global = i % num_cols
        row_local = (i // num_cols) + 2
        col_local = i % num_cols
        # Plot global feature maps
        global_feat_map = global_features[0, i, :, :].detach().cpu().numpy()
        axs[row_global, col_global].imshow(global_feat_map, cmap='hot')
        axs[row_global, col_global].axis('off')

        # Plot local feature maps
        local_feat_map = local_features[0, i, :, :].detach().cpu().numpy()
        axs[row_local, col_local].imshow(local_feat_map, cmap='hot')
        axs[row_local, col_local].axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.01, hspace=0.01)
    plt.tight_layout(pad=0)
    plt.savefig("features_comparison.jpg")
    plt.show()
    print("Feature maps comparison saved as 'features_comparison.jpg'.")


# 调用测试函数
if __name__ == "__main__":
    set_random_seed(42)
    test_model()

