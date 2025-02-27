import json
import os
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm


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

train_json = "/hd2/wangzichuan/Evalmuse/dataset/structure/train_info.json"
img_folder = "/hd2/wangzichuan/Evalmuse/dataset/structure/train/images"
with open(train_json, 'r', encoding='utf-8') as file:
    train_data = json.load(file)
print(len(train_data))

with open('processed_train.jsonl', 'a+', encoding='utf-8') as output_file:
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
        for bbox in bboxes:
            if len(bbox) == 0:
                continue
            else:
                for box in bbox:
                    bbox_type = box['bbox_type']
                    box_info = box['bbox'] 
                    if bbox_type == 1:
                        s_bbox = normalize_coordinates([box_info[0]['x'],box_info[0]['y'],box_info[1]['x'],box_info[1]['y']],width,height)
                    elif bbox_type == 2:
                        points = np.array(box_info)
                        x, y, w, h = cv2.boundingRect(points)
                        x1 = int(x + w * 0.03)
                        y1 = int(y + h * 0.03)
                        x2 = int(x + w * 0.97)
                        y2 = int(y + h * 0.97)
                        s_bbox = normalize_coordinates([x1, y1, x2, y2], width, height)
                    structure_bboxes.append(s_bbox)

        new_item = {"id": str(id), "width": width, "height": height, "image": img_path, "conversations": []}
        new_item["conversations"].append({"from": "human", "value": f"<image>\nThis image is generated from the following prompt: '{prompt}'. How would you evaluate the quality of this image from a structural perspective? Please respond with a single word."})
        new_item["conversations"].append({"from": "gpt", "value": get_score_rate(mos).capitalize() + "."})
        new_item["conversations"].append({"from": "human","value": "Please detect all the areas with structural issues and mark their positions."})
        conversation = {"from": "gpt", "value": f"Sure, I will detect all the areas with structural issues and mark their positions.\n"}
        for b in structure_bboxes:
            conversation["value"] += f"<box>{[b]}</box>\n"
        new_item["conversations"].append(conversation)

        output_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')
