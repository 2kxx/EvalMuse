from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
import json
from tqdm import tqdm
import os
import torch.nn.functional as F
import re
import numpy as np
import pickle


def back_coordinates(box, image_width, image_height):
    x1, y1, x2, y2 = box
    back_box = [
        round((x1 / 1000) * image_width),
        round((y1 / 1000) * image_height),
        round((x2 / 1000) * image_width),
        round((y2 / 1000) * image_height)
    ]
    return back_box


{'excellent':48920, 'good':15338, 'fair':59444, 'poor':299, 'bad':17222}
{'yes':9583, 'no':2917}

model = '/hd2/wangzichuan/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/evalmuse/merged_s_internvl2_5_8b_dynamic_res_2nd_finetune_lora/checkpoint-8400'
# model = '/hd2/wangzichuan/InternVL/pretrained/InternVL2_5-8B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8, output_logits='generation', logprobs=5)
img_folder = "/hd2/wangzichuan/Evalmuse/dataset/structure/val/images"
test_json = "/hd2/wangzichuan/Evalmuse/dataset/structure/val_info.json"
with open(test_json, 'r', encoding='utf-8') as f:
    test_data = json.load(f)   

results = {}
results_inverse = {}
for index, (key, value) in enumerate(tqdm(test_data.items())):
    # if index == 1:
    #     break
    result = {}
    result_inverse = {}
    img_path = os.path.join(img_folder, key+'.jpg')
    prompt = value['prompt_en']
    image = load_image(img_path)
    width, height = image.size
    
    q_s = f"This image is generated from the following prompt: '{prompt}'. How would you evaluate the quality of this image from a structural perspective? Please respond with a single word."
    # q_s = f"How would you evaluate the quality of this image from a structural perspective? Please respond with a single word."
    sess = pipe.chat((q_s, image), gen_config=gen_config)
    logits = sess.response.logits
    probs = F.softmax(logits, dim=1)
    score = (5*probs[0][48920] + 4*probs[0][15338] + 3*probs[0][59444] + 2*probs[0][299] + 1*probs[0][17222]).item()
    
    q_l = "Please detect all the areas with structural issues and mark their positions."
    sess = pipe.chat(q_l, session=sess, gen_config=gen_config)
    response_text = sess.response.text
    # print(response_text)
    box_pattern = re.compile(r'\[\[([0-9, ]+)\]\]')
    boxes = []
    matches = box_pattern.findall(response_text)
    for match in matches:
        box = list(map(int, match.split(", ")))
        box = back_coordinates(box, width, height)
        boxes.append(box)
    map_matrix = np.zeros((height, width), dtype=np.uint8)
    map_matrix_inverse = np.zeros((height, width), dtype=np.uint8)
    for box in boxes:
        # print(box)
        x1, y1, x2, y2 = box
        map_matrix[y1:y2, x1:x2] = 1
        map_matrix_inverse[x1:x2, y1:y2] = 1
    
    # print(map_matrix.max(), map_matrix.min())

    result['score'] = score
    result['pred_area'] = map_matrix
    results[key] = result

    result_inverse['score'] = score
    result_inverse['pred_area'] = map_matrix_inverse
    results_inverse[key] = result_inverse


with open("/hd2/wangzichuan/Evalmuse/EvalMuse-Structure-main/submission/output.pkl", "wb+") as f:
    pickle.dump(results, f)

with open("/hd2/wangzichuan/Evalmuse/EvalMuse-Structure-main/submission/output_inverse.pkl", "wb+") as f:
    pickle.dump(results_inverse, f)