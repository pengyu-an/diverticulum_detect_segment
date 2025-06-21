import os
import copy
import json
import shutil

def write_yolo_txt_file(txt_file_path, label_seg_x_y_list):
    if not os.path.exists(txt_file_path):
        with open(txt_file_path, "w") as file:
            for element in label_seg_x_y_list:
                file.write(str(element) + " ")
            file.write('\n')
    else:
        with open(txt_file_path, "a") as file:
            for element in label_seg_x_y_list:
                file.write(str(element) + " ")
            file.write('\n')


def read_json(json_path, target_dir):
    with open(json_path, "r", encoding='utf-8') as f:
        json_data = json.load(f)

    for annotation in json_data['annotations']: # 遍历标注数据信息
        category_id = annotation['category_id']
        image_id = annotation['image_id']
        for image in json_data['images']:
            if image['id'] == image_id:
                width = image['width']
                height = image['height']
                txt_file_name = image['file_name'].split('.')[0] + '.txt' # 要保存的对应txt文件名
                break

        segmentation = annotation['segmentation']  # 分割点信息 [[x1,y1,x2,y2,...,xn,yn]]
        seg_x_y_list = [i/width if num%2==0 else i/height for num,i in enumerate(segmentation[0])] # 归一化图像分割点信息
        # label_seg_x_y_list = seg_x_y_list[:]  # 修改新列表会更改原始列表
        label_seg_x_y_list = copy.deepcopy(seg_x_y_list)  # 修改新列表不会更改原始列表
        label_seg_x_y_list.insert(0, category_id)  # 图像类别与分割点信息 [label,x1,y1,x2,y2,...,xn,yn]

        # 写txt文件
        txt_file_path = os.path.join(target_dir, txt_file_name)
        write_yolo_txt_file(txt_file_path, label_seg_x_y_list)


if __name__=="__main__":
    dataset_root = 'dataset'
    label_dir = 'dataset/labels'
    
    # 处理训练集和测试集
    for split in ["train", "test"]:
        json_path = os.path.join(label_dir, f"{split}.json")
        target_dir = os.path.join('dataset/labels_segment', split)
        os.makedirs(target_dir, exist_ok=True)
    
        read_json(json_path, target_dir)
