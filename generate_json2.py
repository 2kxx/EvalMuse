import json
import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from pycocotools import mask


def normalize_coordinates(box, image_width, image_height):
    x1, y1, x2, y2 = box
    normalized_box = [
        round((x1 / image_width) * 1000),
        round((y1 / image_height) * 1000),
        round((x2 / image_width) * 1000),
        round((y2 / image_height) * 1000)
    ]
    return normalized_box

def get_score_rate(score):
    if 1 <= score < 1.5:
        return "bad"
    elif 1.5 <= score < 2.5:
        return "poor"
    elif 2.5 <= score < 3.5:
        return "fair"
    elif 3.5 <= score < 4.5:
        return "good"
    elif 4.5 <= score <= 5:
        return "excellent"
    else:
        print("score error!")

train_json = "/hd2/tangzhenchen/dataset/EvalMuse-Structure/train_info.json"
img_folder = "/hd2/wangzichuan/Evalmuse/dataset/structure/train/images"
with open(train_json, 'r', encoding='utf-8') as file:
    train_data = json.load(file)
print(len(train_data))

with open('processed_train.jsonl', 'w+', encoding='utf-8') as output_file:
    id = 0
    for key, value in tqdm(train_data.items()):
        id = id + 1
        img_path = os.path.join(img_folder, key+'.jpg')
        img = Image.open(img_path)
        width, height = img.size
        prompt = value['prompt_en']
        bboxes = value['bbox_info']
        mos = value['mos']
        structure_bboxes = []
        final_mask = np.zeros((height, width), dtype=np.uint8)
        for bbox in bboxes:
            if len(bbox) == 0:
                continue
            for box in bbox:
                bbox_type = box['bbox_type']
                box_info = box['bbox']

                if bbox_type == 1:
                    # 类型1：矩形框，填值为1
                    x1, y1 = int(box_info[0]['x']), int(box_info[0]['y'])
                    x2, y2 = int(box_info[1]['x']), int(box_info[1]['y'])
                    cv2.rectangle(final_mask, (x1, y1), (x2, y2), color=1, thickness=-1)

                elif bbox_type == 2:
                    # 类型2：多边形segmentation
                    if len(box_info) < 3:
                        print("Skipping invalid bbox with less than 3 points:", box_info)
                        continue  # 跳过该数据
                    flattened_box_info = [coord for point in box_info for coord in point]

                    # 再包装一层 list（符合 frPyObjects 的输入格式）
                    box_info_for_rle = [flattened_box_info]
                    rle = mask.frPyObjects(box_info_for_rle, height, width)
                    m = mask.decode(rle)
                    if len(m.shape) > 2:
                        m = np.sum(m, axis=2)
                    m = m.astype(np.uint8)
                    final_mask[m > 0] = 1  # segmentation优先覆盖

        new_item = {"id": str(id), "width": width, "height": height, "image": img_path, "conversations": [], "mask": final_mask.tolist()}
        new_item["conversations"].append({"from": "human", "value": f"<image>\nThis image is generated from the following prompt: '{prompt}'. How would you evaluate the quality of this image from a structural perspective? Please respond with a single word."})
        new_item["conversations"].append({"from": "gpt", "value": get_score_rate(mos).capitalize() + "."})
        new_item["conversations"].append({"from": "human", "value": "Please detect all the areas with structural issues and mark their positions."})
        conversation = {"from": "gpt", "value": f"Sure, I will detect all the areas with structural issues and mark their positions. [SEG1][SEG2][SEG3]\n"}

        output_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')
