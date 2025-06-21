import os
import json

def main():
    dataset_root = "dataset_segment"
    labels_dir = os.path.join(dataset_root, "labels")
    with open(os.path.join(labels_dir, "train.json")) as f:
        train_data = json.load(f)
    
    # 创建类别ID映射（COCO ID -> YOLO连续ID）
    categories = sorted(train_data['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories]
    dataset_path = os.path.abspath(dataset_root).replace('\\', '/')


    seg_yaml_content = f"""path: {dataset_path}
train: images/train
val: images/val
test: images/test

nc: {len(class_names)}  # Number of classes

names:
  0: qishi  # name of class 0
"""

    with open(os.path.join("./train_models/yolov8", "yolo_seg_segment.yaml"), "w") as f:
        f.write(seg_yaml_content)


if __name__ == "__main__":
    main()