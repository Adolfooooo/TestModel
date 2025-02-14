import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

class FeatureExtractor():
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.features = []
        self.hooks = []
        
        # 注册钩子
        for layer_name in target_layers:
            layer = dict([*self.model.named_modules()])[layer_name]
            hook = layer.register_forward_hook(self._get_features)
            self.hooks.append(hook)
    
    def _get_features(self, module, input, output):
        self.features.append(output)
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
    
    def get_features(self, x):
        self.features = []
        _ = self.model(x)
        return self.features

def visualize_feature_maps(feature_maps, num_channels=16, figsize=(15, 15), pic_name="feature_map", is_batch=True):
    """可视化特征图
    Args:
        feature_maps: 特征图张量 (batch_size, channels, height, width)
        num_channels: 要显示的通道数量
        figsize: 图像大小
    """
    # 确保feature_maps是CPU上的numpy数组
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.detach().cpu().numpy()
    
    # 获取第一个样本的特征图
    if is_batch:
        first_feature_maps = feature_maps[0]
    else:
        first_feature_maps = feature_maps
    
    # 确定要显示的通道数量
    n_channels = min(num_channels, first_feature_maps.shape[0])
    
    # 计算子图的行列数
    nrows = int(np.sqrt(n_channels))
    ncols = int(np.ceil(n_channels / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    # 确保 axes 始终是数组，因为当只有一个子图时（nrows=ncols=1），返回的是单个 Axes 对象而不是数组
    axes = np.array(axes)
    axes = axes.ravel()
    
    for idx in range(n_channels):
        feature_map_channel = first_feature_maps[idx]
        
        # 归一化到[0,1]区间
        feature_map_channel = (feature_map_channel - feature_map_channel.min()) / (feature_map_channel.max() - feature_map_channel.min() + 1e-8)
        
        axes[idx].imshow(feature_map_channel, cmap='viridis')
        axes[idx].axis('off')
        axes[idx].set_title(f'Channel {idx}')

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{pic_name}.png")
    plt.close()


def visualize_non_channel_feature_maps(feature_maps, figsize=(15, 15), pic_name="label", is_batch=True):
    """可视化特征图
    Args:
        feature_maps: 特征图张量 (batch_size, channels, height, width)
        num_channels: 要显示的通道数量
        figsize: 图像大小
    """
    # 确保feature_maps是CPU上的numpy数组
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.detach().cpu().numpy()
    
    # 获取第一个样本的特征图
    if is_batch:
        first_feature_maps = feature_maps[0]
    else:
        first_feature_maps = feature_maps
        
    # 确定要显示的通道数量
    n_channels = 1
    
    # 计算子图的行列数
    nrows = int(np.sqrt(n_channels))
    ncols = int(np.ceil(n_channels / nrows))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    # 确保 axes 始终是数组，因为当只有一个子图时（nrows=ncols=1），返回的是单个 Axes 对象而不是数组
    axes = np.array(axes)
    axes = axes.ravel()
    

    feature_map_channel = first_feature_maps
    
    # 归一化到[0,1]区间
    feature_map_channel = (feature_map_channel - feature_map_channel.min()) / (feature_map_channel.max() - feature_map_channel.min() + 1e-8)
    
    axes[0].imshow(feature_map_channel, cmap='viridis')
    axes[0].axis('off')
    axes[0].set_title(f'Channel {0}')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'{pic_name}.png')
    plt.close()


# 使用示例
def example_usage():
    # 假设我们有一个预训练的模型
    import torchvision.models as models
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # 选择要可视化的层
    target_layers = ['layer1.1.conv2', 'layer2.1.conv2']
    
    # 创建特征提取器
    extractor = FeatureExtractor(model, target_layers)
    
    # 创建示例输入
    input_tensor = torch.randn(1, 3, 224, 224)
    
    # 获取特征图
    features = extractor.get_features(input_tensor)
    
    # 可视化每一层的特征图
    for idx, feature_map in enumerate(features):
        print(f"Visualizing {target_layers[idx]} output")
        visualize_feature_maps(feature_map)
    
    # 清除钩子
    extractor.remove_hooks()

# 在实际训练过程中使用示例
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Linear(128 * 112 * 112, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train_with_visualization(model, train_loader, criterion, optimizer, device, layer_names):
    # 创建特征提取器
    extractor = FeatureExtractor(model, layer_names)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播并获取特征图
        features = extractor.get_features(data)
        
        # 定期可视化特征图
        if batch_idx % 100 == 0:
            for idx, feature_map in enumerate(features):
                print(f"Batch {batch_idx}, Layer {layer_names[idx]}")
                visualize_feature_maps(feature_map)
        
        # 正常的训练步骤
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # 训练结束后移除钩子
    extractor.remove_hooks()
