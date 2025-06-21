import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO
import torch
import torch
from typing import List

def tta_inference_batch(model, batch_paths, save=False, verbose=False):
    """
    对批量图像路径进行TTA推理
    Args:
        model: 加载好的YOLO模型
        batch_paths: 批量图像路径列表
        save, verbose: 保持原参数传递
    Returns:
        与原始model调用相同格式的结果
    """
    # 1. 原始推理
    orig_results = model(batch_paths, save=save, verbose=verbose)
    
    # 2. 准备水平翻转的图像路径
    flip_paths = []
    for path in batch_paths:
        # 创建临时翻转图像文件路径
        flip_path = f"flip_{path.split('/')[-1]}"
        os.makedirs(os.path.dirname(flip_path), exist_ok=True)  # 确保目录存在
        img = cv2.imread(path)
        flip_img = cv2.flip(img, 1)  # 水平翻转
        cv2.imwrite(flip_path, flip_img)
        flip_paths.append(flip_path)
    
    # 3. 对翻转图像进行推理
    flip_results = model(flip_paths, save=False, verbose=False)
    
    # 4. 处理每张图像的结果
    final_results = []
    for orig_res, flip_res in zip(orig_results, flip_results):
        # 获取原始分割结果
        orig_mask = orig_res.masks.data if orig_res.masks is not None else None
        
        # 获取翻转结果并反转回原始方向
        if flip_res.masks is not None:
            flip_mask = torch.flip(flip_res.masks.data, [2])  # 水平翻转回来
        else:
            flip_mask = None
        
        # 5. 融合分割结果
        if orig_mask is not None and flip_mask is not None:
            if orig_mask.shape[0] != flip_mask.shape[0]:
                # 选择一个合适的策略来调整大小，例如裁剪或填充
                # 这里我们选择裁剪flip_mask以匹配orig_mask的大小
                flip_mask = flip_mask[:orig_mask.shape[0]]

            # 现在两个张量的大小应该匹配，可以进行逐元素操作
            fused_mask = (orig_mask + flip_mask) / 2.0
            
            orig_res.masks.data = fused_mask  # 更新分割结果
        
        # 保留原始检测结果，只更新分割
        final_results.append(orig_res)
        
        # 清理临时文件
        # os.remove(flip_path)
    
    return final_results


def parse_yolo_seg_file(txt_path, img_width, img_height):
    """
    解析YOLO格式的分割标注文件
    返回: 多边形点列表 [ [ [x1,y1], [x2,y2], ... ], ... ]
    """
    polygons = []
    if not os.path.exists(txt_path):
        return polygons
    
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            # 跳过空行
            if not parts:
                continue
            # 提取多边形点（跳过类别ID）
            points = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            points = points.reshape(-1, 2)
            # 归一化坐标 -> 绝对坐标
            points[:, 0] *= img_width
            points[:, 1] *= img_height
            polygons.append(points.astype(np.int32))
    return polygons


def create_mask_from_polygons(polygons, width, height):
    """从多边形列表创建二值掩码"""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [poly], color=1)
    return mask


def dice_coefficient(mask_pred, mask_gt):
    """计算Dice系数"""
    intersection = np.logical_and(mask_pred, mask_gt)
    numerator = 2 * np.sum(intersection)
    denominator = np.sum(mask_pred) + np.sum(mask_gt)
    
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def save_yolo_seg_results(results, save_dir, img_name):
    """保存YOLO格式的分割结果"""
    txt_path = os.path.join(save_dir, f"{Path(img_name).stem}.txt")
    
    with open(txt_path, 'w') as f:
        for result in results:
            
            # f.write("1111")
            
            # 每个结果对应一张图片的预测
            if result.masks is None:
                continue
                
            for mask, cls in zip(result.masks.xy, result.boxes.cls):
                # 转换为归一化坐标
                normalized_mask = mask.copy()
                normalized_mask[:, 0] /= result.orig_shape[1]  # 宽度归一化
                normalized_mask[:, 1] /= result.orig_shape[0]  # 高度归一化
                
                # 写入文件: class_id x1 y1 x2 y2 ...
                line = [int(cls.item())] + normalized_mask.flatten().tolist()
                f.write(" ".join(map(str, line)) + "\n")
                # f.write("1111")

# =============== 配置区域 ===============
model_weights = "./outputs/segment/yolov8n-seg-2251/weights/best.pt"       # 训练好的模型权重
# model_weights = "./outputs/segment/yolov8n-seg3/weights/best.pt" 
images_dir = "./dataset/labels/test"      # 测试集图片目录
labels_dir = "./dataset/labels/test"      # 真实标签目录
preds_dir = "./outputs/predictions"       # 模型预测结果目录

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 自动选择设备

# ======================================

# 创建预测结果目录
os.makedirs(preds_dir, exist_ok=True)

# 加载训练好的模型
print(f"🚀 加载模型: {model_weights}")
model = YOLO(model_weights)
model.to(device)

# 第一步：使用模型进行预测并保存结果
print("🔍 开始预测测试集图片...")
image_files = [f for f in os.listdir(images_dir) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# 批量预测提高效率
batch_size = 8  # 根据GPU内存调整
for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i:i+batch_size]
    batch_paths = [os.path.join(images_dir, f) for f in batch_files]
    
    # 原始调用
    # results = model(batch_paths, save=False, verbose=False)

    # 改为TTA调用
    results = tta_inference_batch(model, batch_paths, save=False, verbose=False)

    # print(results[0])
    
    # 保存预测结果
    for img_path, result in zip(batch_paths, results):
        save_yolo_seg_results([result], preds_dir, os.path.basename(img_path))
    
    print(f"📊 已处理: {min(i+batch_size, len(image_files))}/{len(image_files)} 张图片")

# 第二步：计算Dice系数
print("\n🧮 开始计算Dice系数...")
total_dice = 0.0
num_images = 0
dice_results = []

# 遍历测试集图片
for img_file in image_files[:24]:
    img_path = os.path.join(images_dir, img_file)
    
    # 读取图像获取尺寸
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 无法读取图片: {img_file}, 跳过")
        continue
        
    img_height, img_width = img.shape[:2]
    
    # 构建对应标签路径
    txt_name = Path(img_file).stem + ".txt"
    gt_txt_path = os.path.join(labels_dir, txt_name)
    pred_txt_path = os.path.join(preds_dir, txt_name)

    # 解析多边形
    gt_polygons = parse_yolo_seg_file(gt_txt_path, img_width, img_height)
    pred_polygons = parse_yolo_seg_file(pred_txt_path, img_width, img_height)
    
    # 创建掩码
    gt_mask = create_mask_from_polygons(gt_polygons, img_width, img_height)
    pred_mask = create_mask_from_polygons(pred_polygons, img_width, img_height)
    
    print(gt_mask[0][1], pred_mask[0][1])
    
    # 计算Dice系数
    dice = dice_coefficient(pred_mask, gt_mask)
    total_dice += dice
    num_images += 1
    dice_results.append((img_file, dice))
    
    print(f"📈 {img_file}: Dice = {dice:.4f}")

# 计算平均Dice
if num_images > 0:
    mean_dice = total_dice / num_images
    print("\n" + "="*50)
    print(f"✅ 处理完成! 共处理 {num_images} 张图片")
    print(f"🎯 平均 Dice 系数: {mean_dice:.4f}") 
    
    # 保存详细结果
    results_csv = os.path.join(preds_dir, "dice_results.csv")
    with open(results_csv, 'w') as f:
        f.write("image,dice\n")
        for img_file, dice in dice_results:
            f.write(f"{img_file},{dice:.4f}\n")
    print(f"详细结果保存至: {results_csv}")
else:
    print("❌ 未处理任何图片，请检查路径设置")