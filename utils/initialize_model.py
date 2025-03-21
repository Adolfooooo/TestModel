import torch
import torch.nn as nn
import math

class ModelInitializer:
    """用于初始化PyTorch模型参数的工具类"""
    
    @staticmethod
    def xavier_uniform_init(model):
        """Xavier均匀分布初始化"""
        for param in model.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
                
    @staticmethod
    def xavier_normal_init(model):
        """Xavier正态分布初始化"""
        for param in model.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_normal_(param)
                
    @staticmethod
    def kaiming_uniform_init(model, mode='fan_in', nonlinearity='relu'):
        """Kaiming均匀分布初始化"""
        for param in model.parameters():
            if len(param.shape) >= 2:
                nn.init.kaiming_uniform_(param, mode=mode, nonlinearity=nonlinearity)
                
    @staticmethod
    def kaiming_normal_init(model, mode='fan_in', nonlinearity='relu'):
        """Kaiming正态分布初始化"""
        for param in model.parameters():
            if len(param.shape) >= 2:
                nn.init.kaiming_normal_(param, mode=mode, nonlinearity=nonlinearity)
    
    @staticmethod
    def custom_init(model, mean=0.0, std=0.02):
        """自定义正态分布初始化"""
        for param in model.parameters():
            if len(param.shape) >= 2:
                nn.init.normal_(param, mean=mean, std=std)

# 示例网络模型
class ExampleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ExampleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# 使用示例
def initialize_model():
    # 创建模型实例
    model = ExampleModel(input_dim=784, hidden_dim=256, output_dim=10)
    
    # 选择初始化方法
    initializer = ModelInitializer()
    
    # 使用Xavier均匀分布初始化
    initializer.xavier_uniform_init(model)
    
    # 或者使用Kaiming正态分布初始化
    # initializer.kaiming_normal_init(model)
    
    # 或者使用自定义初始化
    # initializer.custom_init(model, mean=0.0, std=0.01)
    
    return model

# if __name__ == "__main__":
#     # 初始化模型
#     model = initialize_model()
    
#     # 打印模型参数统计信息
#     for name, param in model.named_parameters():
#         print(f"Layer: {name}")
#         print(f"Shape: {param.shape}")
#         print(f"Mean: {param.data.mean().item():.4f}")
#         print(f"Std: {param.data.std().item():.4f}")
#         print("-" * 50)