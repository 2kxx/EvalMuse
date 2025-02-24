from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
import json
from tqdm import tqdm
import os
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model = '/hd2/tangzhenchen/project/EvalMuse-internvl/internvl_chat/work_dirs/internvl_chat_v2_5/internvl2_5_8b_dynamic_res_2nd_finetune_lora_coco_merge2'
# model = '/hd2/wangzichuan/InternVL/pretrained/InternVL2_5-8B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8, output_logits='generation', logprobs=5)
img_folder = "/hd2/tangzhenchen/dataset/EvalMuse/images/"
test_json = "/hd2/tangzhenchen/dataset/EvalMuse/test.json"
with open(test_json, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

results = []
for index, data in enumerate(tqdm(test_data)):
    prompt = data['prompt']
    img_path = os.path.join(img_folder, data['img_path'])
    image = load_image(img_path)

    q_a = f"This image is generated from the following prompt: '{prompt}'. On a scale from 0 to 100, how would you rate the degree of alignment between the image and the prompt? Please provide a number within this range."
    sess = pipe.chat((q_a, image), gen_config=gen_config)
    score = sess.response.text
    score = score.rstrip('.')
    score = float(score)
    score_a = score / 20

    element_scores = {}
    for key, score in data["element_score"].items():
        last_left = key.rfind('(')
        last_right = key.rfind(')')
        category = key[last_left + 1:last_right].lower()
        object = key[:last_left].rstrip()

        if '-' in category:
            category = category.split('-')[0]
        if '/' in category:
            category = category.split('/')[0]

        if category in ['object', 'human', 'animal', 'food', 'location']:
            question = f"Is '{object}' visible in the image? Please rate its presence on a scale from 0 to 30."
        elif category in ['activity', 'attribute', 'color', 'material', 'shape']:
            question = f"Is the {category} '{object}' visible in the image? Please rate its presence on a scale from 0 to 30."
        elif category == 'counting':
            question = f"Is the quantity '{object}' visible in the image? Please rate its presence on a scale from 0 to 30."
        elif category == 'spatial':
            question = f"Is the spatial relationship '{object}' visible in the image? Please rate its presence on a scale from 0 to 30."
        elif category == 'other':
            question = f"Is '{object}' visible in the image? Please rate its presence on a scale from 0 to 30."
        else:
            print(category, object)
            question = f"Is '{object}' visible in the image? Please rate its presence on a scale from 0 to 30."

        q_e = question
        sess = pipe.chat(q_e, session=sess, gen_config=gen_config)
        score = sess.response.text
        score = score.rstrip('.')
        score = float(score)
        score_e = score / 30

        element_scores[key] = score_e

    result = {
        "prompt_id": data['prompt_id'],
        "prompt": prompt,
        "type": data['type'],
        "img_path": data['img_path'],
        "total_score": score_a,
        "element_score": element_scores,
        "promt_meaningless": data['promt_meaningless'],
        "split_confidence": data['split_confidence'],
        "attribute_confidence": data['attribute_confidence'],
        "fidelity_label": data['fidelity_label']
    }
    results.append(result)
    # with open('/hd2/wangzichuan/Evalmuse/submission/output.json', 'w+') as f:
    #     json.dump(results, f, indent=4)

with open('./output2.json', 'w+') as f:
    json.dump(results, f, indent=4)
