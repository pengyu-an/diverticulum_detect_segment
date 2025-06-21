# diverticulum_detect_segment

## 数据准备

首先确保 './dataset' 文件夹存在。假设项目根目录为 root，那么先运行下面的代码，将 label 标签文件转换为 yolo 可以读取的 yaml 文件。

```bash
cd root
python scripts/gen_yaml.py
```

再运行下面的代码，将 './dataset/label' 中两个 json 格式存储的标签转换为检测任务和分割任务的标签。

```bash
python scripts/coco2yolo.py
python scripts/coco2label.py
```

随后，可以将 './dataset/labels_detect' 和 './dataset/labels_segment' 分开，并从训练集中抽出一定比例的数据和标签作为验证集。最终数据存放结构如下：

```bash
diverticulum_detect_segment/
├── dataset_detect/
│   ├── images
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── labels
│       ├── test
│       ├── train
│       └── val
├── dataset_segment/
│   ├── images
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── labels
│       ├── test
│       ├── train
│       └── val
......
```

## 运行代码

训练 YOLO 模型：

```bash
cd root
python main.py
```

测试 YOLO 模型的分割效果，计算dice值：

```bash
cd root
python scripts/dice_cal.py
```

使用 TTA 方法测试 YOLO 模型的分割效果，计算dice值：

```bash
cd root
python scripts/dice_tta.py
```

