import os
import random
import logging
import argparse
import numpy as np
os.path.join("./train_models/yolov12/ultralytics")
import torch
from ultralytics import YOLO
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def set_seed(seed=114514):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(args):
    print(args.model)
    if args.model == 'yolo-seg':
        model = YOLO("./train_models/yolov8/yolov8-seg.yaml")  # 加载 YOLOv8 分割模型配置
        model.load('./train_models/yolov8/yolov8n-seg.pt').to(args.device)  # 加载预训练权重
        model.train(
            data='./train_models/yolov8/yolo_seg_segment.yaml',  # 数据集 yaml 路径
            imgsz=args.img_size,  # 输入图像大小
            epochs=args.epochs,  # 训练轮数
            batch=args.batch_size,  # 批次大小
            lr0=args.lr,  # 初始学习率
            optimizer=args.optimizer,  # 优化器选择
            workers=args.num_workers,  # 数据加载工作线程数
            scale=0.5,  # 图像缩放因子，影响数据增强中的尺度变化
            mosaic=1.0,  # 启用 Mosaic 数据增强，值为 1.0 表示完全启用
            mixup=0.2,  # 设置 MixUp 数据增强的混合比例
            copy_paste=0.2,  # 设置 Copy-Paste 数据增强的概率，对分割任务有显著影响
            device='0' if torch.cuda.is_available() else 'cpu',  # 设备选择，使用 GPU 或 CPU
            project='./outputs/segment',  # 指定保存训练结果的根目录
            name=f'yolov8n-seg-{datetime.now().timestamp()}',  # 指定保存训练结果的子目录名称
            mask_ratio=4,  # 分割掩码比例，影响掩码下采样率
            overlap_mask=True,  # 处理重叠掩码
        )
        
    torch.save(model.state_dict(), os.path.join(args.output_dir, f'{args.model}.pt'))
    logger.info(f'Model saved to {args.output_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolo-seg', 
                      choices=['maskrcnn', 'yolo', 'yolo-seg'],
                      help='Model architecture selection')
    parser.add_argument('--optimizer', type=str, default='SGD',
                      choices=['Adamw', 'Adam', 'SGD'])
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--img_size', type=int, default=640, help='Input image size for training')
    parser.add_argument('--num_workers', type=int, default=32, help='Number of workers for data loading')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=2251, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f'Using device: {args.device}')
    logger.info(f'Training with model: {args.model}')
    train(args)
    
if __name__ == "__main__":
    main()
