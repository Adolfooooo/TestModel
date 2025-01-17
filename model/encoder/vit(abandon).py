import torch
from torchvision.models import vit_b_16
from collections import OrderedDict

def load_custom_vit(checkpoint_path, num_classes=1000, device='cuda'):
    """
    加载自定义预训练权重的ViT模型
    
    参数:
    - checkpoint_path: 预训练权重文件路径(.pth文件)
    - num_classes: 输出类别数量
    - device: 运行设备
    
    返回:
    - model: 加载好权重的模型
    """
    # 初始化模型
    model = vit_b_16(num_classes=num_classes)
    
    # 加载权重文件
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理不同类型的权重文件
    if isinstance(checkpoint, OrderedDict) or isinstance(checkpoint, dict):
        # 如果直接是状态字典
        state_dict = checkpoint
    else:
        # 如果权重文件包含其他信息（如训练状态等）
        state_dict = checkpoint.get('state_dict', checkpoint)
    
    # 处理权重键名不匹配的情况
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # 移除"module."前缀（如果存在）
        if k.startswith('module.'):
            name = k[7:]  # 去除 'module.' 前缀
        else:
            name = k
        new_state_dict[name] = v
    
    # 尝试加载权重
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("成功加载全部权重")
    except RuntimeError as e:
        # 如果严格加载失败，打印错误信息并尝试非严格加载
        print(f"严格加载失败: {e}")
        print("尝试非严格加载...")
        model.load_state_dict(new_state_dict, strict=False)
        print("成功完成非严格加载")
    
    # 将模型移动到指定设备
    model = model.to(device)
    model.eval()
    
    return model

def verify_model(model, input_size=(3, 224, 224), device='cuda'):
    """
    验证模型是否正确加载并可以进行推理
    
    参数:
    - model: 加载好的模型
    - input_size: 输入张量的大小
    - device: 运行设备
    """
    print("验证模型...")
    # 创建随机输入
    x = torch.randn(1, *input_size).to(device)
    
    try:
        with torch.no_grad():
            output = model(x)
        print(f"模型验证成功！输出形状: {output.shape}")
        return True
    except Exception as e:
        print(f"模型验证失败: {e}")
        return False

# 使用示例
if __name__ == "__main__":
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载模型
    checkpoint_path = "path/to/your/checkpoint.pth"
    model = load_custom_vit(
        checkpoint_path=checkpoint_path,
        num_classes=1000,  # 根据你的模型调整
        device=device
    )
    
    # 验证模型
    verify_model(model, device=device)
    
    # 使用模型进行推理
    def inference(model, input_tensor):
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        return output