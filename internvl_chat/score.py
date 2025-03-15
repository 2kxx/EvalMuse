import copy

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
import json
from tqdm import tqdm
import torch
import os
import torch.nn.functional as F
from transformers import AutoModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

{'excellent':48920, 'good':15338, 'fair':59444, 'poor':299, 'bad':17222}
{'yes':9583, 'no':2917}



model_path = '/hd2/tangzhenchen/project/EvalMuse-internvl/internvl_chat/work_dirs/internvl_chat_v2_5/internvl2_5_8b_dynamic_res_2nd_finetune_lora_coco_merge2-6000'
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()

deepmlp1 = copy.deepcopy(model.deepmlp1)
deepmlp2 = copy.deepcopy(model.deepmlp2)
# score_hidden_fcs = copy.deepcopy(model.score_hidden_fcs)
del model
torch.cuda.empty_cache()

pipe = pipeline(model_path, backend_config=TurbomindEngineConfig(session_len=8192))
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8, output_logits='generation', output_last_hidden_state='generation', skip_special_tokens=False, spaces_between_special_tokens=False, logprobs=5)
img_folder = "/hd2/tangzhenchen/dataset/EvalMuse/images/"
test_json = "/hd2/tangzhenchen/dataset/EvalMuse/test.json"

# 加载测试数据
with open(test_json, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 加载或初始化结果和保存的进度
output_file = './output6000.json'
checkpoint_file = './test_checkpoint6000.json'

# 尝试从文件加载上次的进度
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        checkpoint = json.load(f)
    start_index = checkpoint.get('start_index', 0)
    results = checkpoint.get('results', [])
else:
    start_index = 0
    results = []

pattern1 = [9202, 1811, 6776, 7989, 4028]
pattern2 = [9970, 1038]

for index, data in enumerate(tqdm(test_data[start_index:], desc="Processing")):
    prompt = data['prompt']
    img_path = os.path.join(img_folder, data['img_path'])
    image = load_image(img_path)

    q_a = f"This image is generated from the following prompt: '{prompt}'. Evaluate the image-text alignment for this prompt and give a score."
    sess = pipe.chat((q_a, image), gen_config=gen_config)
    embedding = sess.response.last_hidden_state.to('cuda')
    print(sess.response.text)
    token_ids = sess.response.token_ids
    idx1 = 0
    idx2 = 0
    for idx, token in enumerate(token_ids):
        if token == 92553:
            idx1 = idx
        elif token == 92554:
            idx2 = idx
    embedding = torch.cat([embedding[idx1], embedding[idx2]], dim=-1)
    # embedding = score_hidden_fcs[0](embedding[idx])
    score_a = deepmlp1(embedding).to(torch.float32).item()

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

        question = f"Assess the coherence of the image and prompt related to '{object}'. Output a score."
        q_e = question
        sess = pipe.chat(q_e, session=sess, gen_config=gen_config)
        print(sess.response.text)
        embedding = sess.response.last_hidden_state.to('cuda')
        token_ids = sess.response.token_ids
        idx1 = 0
        idx2 = 0
        for idx, token in enumerate(token_ids):
            if token == 92555:
                idx1 = idx
            elif token == 92556:
                idx2 = idx
        embedding = torch.cat([embedding[idx1], embedding[idx2]], dim=-1)
        score_e = deepmlp2(embedding).to(torch.float32).item()

        element_scores[key] = score_e / 6

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

    # 每500个数据保存一次
    if (index + 1) % 500 == 0:
        with open(output_file, 'w+') as f:
            json.dump(results, f, indent=4)

        checkpoint = {
            'start_index': start_index + index + 1,
            'results': results
        }

        with open(checkpoint_file, 'w+') as f:
            json.dump(checkpoint, f, indent=4)

# 最后保存所有结果
with open(output_file, 'w+') as f:
    json.dump(results, f, indent=4)