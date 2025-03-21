from torch.utils.data import DataLoader

def get_dataloader(dataset, batch_size=4, shuffle=True, num_workers=4):
    """
    创建ACDC数据集的DataLoader
    Args:
        dataset: 
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的进程数
    Returns:
        DataLoader对象
    """
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader