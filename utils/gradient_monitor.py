import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import os
from utils.global_variable import global_variable_set_dict,global_variable_get

class GradientMonitor:
    def __init__(self, model, log_dir='runs/gradient_monitor'):
        """
        初始化梯度监控器
        Args:
            model: PyTorch 模型
            log_dir: TensorBoard 日志目录
        """
        self.model = model
        self.writer = SummaryWriter(log_dir)
        self.grad_stats = defaultdict(list)
        self.hooks = []
        self.step = 0
        self._register_hooks()
        
    def _register_hooks(self):
        """注册梯度钩子到每个参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, name=name: self._grad_hook(grad, name)
                )
                self.hooks.append(hook)
    
    def _grad_hook(self, grad, name):
        """记录梯度统计信息"""
        if grad is not None:
            stats = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.max().item(),
                'min': grad.min().item(),
                'norm': grad.norm().item()
            }
            self.grad_stats[name].append(stats)
        return grad
    
    def step_monitor(self):
        """每个训练步骤后的监控"""
        for name, stats_list in self.grad_stats.items():
            if stats_list:
                stats = stats_list[-1]
                
                # 记录到 TensorBoard
                self.writer.add_scalar(f'gradients/{name}/mean', stats['mean'], self.step)
                self.writer.add_scalar(f'gradients/{name}/std', stats['std'], self.step)
                self.writer.add_scalar(f'gradients/{name}/max', stats['max'], self.step)
                self.writer.add_scalar(f'gradients/{name}/min', stats['min'], self.step)
                self.writer.add_scalar(f'gradients/{name}/norm', stats['norm'], self.step)
                
                # 检查梯度问题
                self._check_gradient_issues(name, stats)
        
        self.step += 1
    
    def _check_gradient_issues(self, name, stats):
        logger = global_variable_get("logger")
        """检查梯度问题并发出警告"""
        if abs(stats['mean']) < 1e-7:
            # print(f"警告: {name} 可能存在梯度消失问题")
            logger.info(f"警告: {name} 可能存在梯度消失问题")
            self.writer.add_scalar(f'gradient_issues/{name}/vanishing', 1, self.step)
        
        if abs(stats['mean']) > 1e2 or abs(stats['max']) > 1e3:
            # print(f"警告: {name} 可能存在梯度爆炸问题")
            logger.info(f"警告: {name} 可能存在梯度爆炸问题")
            self.writer.add_scalar(f'gradient_issues/{name}/exploding', 1, self.step)
    
    def plot_gradient_flow(self):
        """绘制梯度流动图"""
        plt.figure(figsize=(15, 10))
        
        for name, stats_list in self.grad_stats.items():
            means = [s['mean'] for s in stats_list]
            plt.plot(means, label=name)
        
        plt.xlabel('Training steps')
        plt.ylabel('Gradient mean')
        plt.title('Gradient Flow')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # 保存到 TensorBoard
        self.writer.add_figure('gradient_flow', plt.gcf(), self.step)
        plt.close()
    
    def plot_gradient_distribution(self):
        """绘制梯度分布图"""
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                plt.figure(figsize=(10, 5))
                plt.hist(param.grad.cpu().numpy().flatten(), bins=50)
                plt.title(f'{name} gradient distribution')
                plt.xlabel('Gradient value')
                plt.ylabel('Count')
                
                # 保存到 TensorBoard
                self.writer.add_figure(f'gradient_distribution/{name}', plt.gcf(), self.step)
                plt.close()
    
    def close(self):
        """清理资源"""
        for hook in self.hooks:
            hook.remove()
        self.writer.close()

# 使用示例
# def training_loop_with_monitoring():
#     model = YourModel()
#     optimizer = torch.optim.Adam(model.parameters())
#     monitor = GradientMonitor(model)
    
#     for epoch in range(num_epochs):
#         for batch in dataloader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             loss.backward()
            
#             # 监控梯度
#             monitor.step_monitor()
            
#             # 每隔一定步数绘制分布图
#             if monitor.step % 100 == 0:
#                 monitor.plot_gradient_distribution()
#                 monitor.plot_gradient_flow()
#             optimizer.step()
#     monitor.close()

# 自定义数据记录器
class CustomGradientLogger:
    def __init__(self, log_dir='gradient_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(os.path.join(log_dir, 'gradient_log.txt'), 'w')
    
    def log_gradient_stats(self, name, stats):
        """记录梯度统计信息到文件"""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"{timestamp} - {name}: {stats}\n"
        self.log_file.write(log_entry)
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()