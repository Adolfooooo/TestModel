import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from utils.pytorch_visualization import visualize_feature_maps, visualize_non_channel_feature_maps

'''
npz: ['img', 'label']
image: [1, 224, 224]
classes: five, and one of it is normal.
'''
class ACDCDataset(Dataset):
    def __init__(self, data_dir, transforms: list=None):
        """
        初始化ACDC数据集
        Args:
            data_dir: 数据集所在目录路径
            transform: 数据转换/增强操作
        """
        self.data_dir = data_dir
        self.transforms = transforms
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 加载.npz文件
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)
        
        # 获取图像和标签
        # image, data: (224,224), (224,224)
        image = data['img']  # 假设npz文件中的图像数据键名为'image'
        label = data['label']  # 假设npz文件中的标签数据键名为'label'
        
        # 转换为torch张量
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).long()
        
        # 如果是灰度图像，添加通道维度
        if len(image.shape) == 2:
            image = image.unsqueeze(0)

        # 应用数据转换
        if self.transforms:
            for transform in self.transforms:
                image = transform(image)

        return image, label



# # 使用示例
# if __name__ == '__main__':
#     # 数据转换
#     transform = None  # 可以根据需要定义transforms
    
#     # 创建数据加载器
#     data_dir = 'path/to/your/acdc/dataset'
#     dataloader = get_acdc_dataloader(
#         data_dir=data_dir,
#         batch_size=4,
#         #transforms=transform
#     )
    
#     # 测试数据加载
#     for images, labels in dataloader:
#         print(f"Batch images shape: {images.shape}")
#         print(f"Batch labels shape: {labels.shape}")
#         break  # 只打印第一个批次的信息