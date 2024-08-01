# -*- coding: UTF-8 -*-
"""
===================================================================================
@author : Leoda
@Date   : 2024/05/20 14:39:40
@Project -> : learn$
==================================================================================
"""
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import os
from networks.segformer import MixVisionTransformer


def load_image(image_path, image_size=512):
    """
    加载并调整图像大小
    Args:
        image_path: 图像路径
        image_size: 调整后的图像大小
    Returns:
        调整大小后的图像tensor
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # 增加batch维度
    return image


def visualize_feature_maps_with_metrics(outs, analysis_results, num_images=1, save_path=None):
    """
    可视化 MixVisionTransformer 中的四个阶段特征图，并在图中添加评价指标
    Args:
        outs: MixVisionTransformer 中的四个阶段输出特征图
        analysis_results: 分析结果，包括均值、方差、熵、稀疏性等指标
        num_images: 选择可视化的图像数量
        save_path: 保存可视化结果的路径
    """
    num_stages = len(outs)
    fig, axs = plt.subplots(1, num_stages * num_images, figsize=(5 * num_stages * num_images, 6))
    axs = axs.ravel()

    for i, feature_maps in enumerate(outs):
        for j in range(num_images):
            img_tensor = feature_maps[j]
            img_tensor = img_tensor[0, :, :].unsqueeze(0)  # 选择第一个通道并保持batch维度

            img_grid = vutils.make_grid(img_tensor, normalize=True, scale_each=True, nrow=1)
            np_img = img_grid.cpu().numpy()
            axs[i * num_images + j].imshow(np.transpose(np_img, (1, 2, 0)), cmap='gray')

            metrics = analysis_results[i][j]
            title = (f'Stage {i + 1}\n'
                     f'Mean={metrics["mean"]:.2e}, Std={metrics["std"]:.2f}\n'
                     f'Entropy={metrics["entropy"]:.2f}')
            axs[i * num_images + j].set_title(title)
            axs[i * num_images + j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def analyze_feature_maps(outs):
    analysis_results = []
    for i, feature_maps in enumerate(outs):
        batch_analysis = []
        for j in range(feature_maps.shape[0]):
            img_tensor = feature_maps[j].cpu().numpy()
            img_tensor_normalized = (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min() + 1e-9)

            mean_val = np.mean(img_tensor)
            std_val = np.std(img_tensor)
            entropy_val = -np.sum(img_tensor_normalized * np.log(img_tensor_normalized + 1e-9))


            batch_analysis.append({
                'mean': mean_val,
                'std': std_val,
                'entropy': entropy_val
            })
        analysis_results.append(batch_analysis)
    return analysis_results

def main():
    image_path = 'E:/project/learn/img/test1.jpg'
    output_path = 'img_out/results4.jpg'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    image = load_image(image_path)
    model = MixVisionTransformer()
    model.eval()

    with torch.no_grad():
        outs = model(image)

    analysis_results = analyze_feature_maps(outs)
    visualize_feature_maps_with_metrics(outs, analysis_results, num_images=1, save_path=output_path)


if __name__ == "__main__":
    main()
