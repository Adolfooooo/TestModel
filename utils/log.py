import logging
import os
from datetime import datetime

def get_logger(name='app', level=logging.INFO, log_dir='logs'):
    """
    创建一个带有控制台和文件输出的logger
    
    Args:
        name: 日志文件基础名称
        level: 日志级别
        log_dir: 日志文件存储目录
        
    Returns:
        logging.Logger: 配置好的logger对象
    """
    # 创建logger对象
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 确保移除所有已存在的handler
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 定义日志格式
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(LOG_FORMAT)
    
    # 创建并配置控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 生成唯一的日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_filename = f"{name}_{timestamp}.log"
    log_file = os.path.join(log_dir, base_filename)
    
    # 创建并配置文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # 记录日志文件创建信息
    logger.info(f"日志文件已创建: {log_file}")
    
    return logger