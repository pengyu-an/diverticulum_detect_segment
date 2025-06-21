import json
import os
from collections import defaultdict
from pathlib import Path


def coco2yolo(json_path, output_dir, category_mapping):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images = {img['id']: img for img in coco_data['images']}
    
    image_annotations = defaultdict(list)
    for ann in coco_data['annotations']:
        image_annotations[ann['image_id']].append(ann)

    for img_id, img_info in images.items():
        annotations = image_annotations.get(img_id, [])
        img_w = img_info['width']
        img_h = img_info['height']
        
        base_name = os.path.splitext(img_info['file_name'])[0]
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        
        with open(txt_path, 'w') as f_txt:
            for ann in annotations:
                if 'bbox' not in ann:
                    continue
                
                yolo_cls = category_mapping[ann['category_id']]
                
                x, y, w, h = ann['bbox']
                x_center = (x + w/2) / img_w
                y_center = (y + h/2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                f_txt.write(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")


def main():
    original_folder = Path('./dataset/label')
    new_folder = Path('./dataset/labels')
    if original_folder.exists():
        original_folder.rename(new_folder)
    dataset_root = "dataset"
    labels_dir = os.path.join(dataset_root, "labels")
    with open(os.path.join(labels_dir, "train.json")) as f:
        train_data = json.load(f)
    
    # 创建类别ID映射（COCO ID -> YOLO连续ID）
    categories = sorted(train_data['categories'], key=lambda x: x['id'])
    category_mapping = {cat['id']: i for i, cat in enumerate(categories)}

    # 处理训练集和测试集
    for split in ["train", "test"]:
        json_path = os.path.join(labels_dir, f"{split}.json")
        output_dir = os.path.join(dataset_root, "labels_detect", split)
        
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping")
            continue
            
        coco2yolo(json_path, output_dir, category_mapping)


if __name__ == "__main__":
    main()