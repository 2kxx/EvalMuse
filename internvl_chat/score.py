from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
import json
from tqdm import tqdm
import os
import torch.nn.functional as F
import argparse



{'excellent':48920, 'good':15338, 'fair':59444, 'poor':299, 'bad':17222}
{'yes':9583, 'no':2917}
{'high':12127, 'moderate':307, 'low':24573, 'no':2917}

parser = argparse.ArgumentParser(description="Run image-text alignment evaluation.")
parser.add_argument("--model", required=True, help="Path to the model checkpoint.")
parser.add_argument("--save_path", required=True, help="Path to save the results.")
args = parser.parse_args()

model = args.model
save_path = args.save_path
# model = '/hd2/wangzichuan/InternVL/pretrained/InternVL2_5-8B'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8, output_logits='generation', logprobs=5)
img_folder = "/hd2/wangzichuan/Evalmuse/dataset/alignment/test/images"
test_json = "/hd2/wangzichuan/Evalmuse/dataset/alignment/test/test.json"
with open(test_json, 'r', encoding='utf-8') as f:
    test_data = json.load(f)   

results = []
with open(save_path, 'w+') as f:
    for index, data in enumerate(tqdm(test_data)):
        # if index == 10:
        #     break
        prompt = data['prompt']
        img_path = os.path.join(img_folder, data['img_path'])
        image = load_image(img_path)
        
        q_a = f"This image is generated from the following prompt: '{prompt}'. How would you rate the alignment of this image with the prompt? Please respond with a single word."
        sess = pipe.chat((q_a, image), gen_config=gen_config)
        logits = sess.response.logits
        probs = F.softmax(logits, dim=1)
        # score_a = (5*probs[0][48920] + 4*probs[0][15338] + 3*probs[0][59444] + 2*probs[0][299] + 1*probs[0][17222]).item()
        score_a_probs = [probs[0][17222].item(), probs[0][299].item(), probs[0][59444].item(), probs[0][15338].item(), probs[0][48920].item()]
        weight_a = [1,2,3,4,5]
        score_a = sum([f * i for f, i in zip(weight_a, score_a_probs)])
        
        element_scores = {}
        score_e_probs = {}
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
                question = f"Is '{object}' present in the image?"
            elif category in ['activity', 'attribute', 'color', 'material', 'shape']:
                question = f"Is the {category} '{object}' present in the image?"
            elif category == 'counting':
                question = f"Is the quantity '{object}' present in the image?"
            elif category == 'spatial':
                question = f"Is the spatial relationship '{object}' present in the image?"
            elif category == 'other':
                question = f"Is '{object}' present in the image?"
            else:
                print(category, object)
                question = f"Is the {category} concept '{object}' present in the image?"
            
            # q_e = question +  " Please provide a response indicating the level of presence: 'High presence', 'Moderate presence', 'Low presence', or 'No presence'."
            q_e = question
            sess = pipe.chat(q_e, session=sess, gen_config=gen_config)
            logits = sess.response.logits
            probs = F.softmax(logits, dim=1)
            # score_e = (1*probs[0][9583] + 0*probs[0][2917]).item()
            # score_e = (0.92*probs[0][12127] + 0.58*probs[0][307] + 0.33*probs[0][24537] + 0.08*probs[0][2917]).item()
            # score_e_prob = [probs[0][2917].item(), probs[0][24537].item(), probs[0][307].item(), probs[0][12127].item()]
            # weight_e = [0.08, 0.33, 0.58, 0.92]
            score_e_prob = [probs[0][2917].item(), probs[0][9583].item()]
            weight_e = [0, 1]
            score_e = sum([f * i for f, i in zip(weight_e, score_e_prob)])
            score_e_probs[key] = score_e_prob

            element_scores[key] = score_e

        result = {
            "prompt_id": data['prompt_id'],
            "prompt": prompt,
            "img_path": data['img_path'],
            "total_score": score_a,
            "element_score": element_scores, 
            "total_score_probs": score_a_probs,
            "element_score_probs": score_e_probs, 
            "promt_meaningless": data['promt_meaningless'], 
            "split_confidence": data['split_confidence'],
            "attribute_confidence": data['attribute_confidence'],
            "fidelity_label": data['fidelity_label']
        }
        # results.append(result)
        json.dump(result, f, indent=4)
        f.write(",\n")
        # with open('/hd2/wangzichuan/Evalmuse/submission/output.json', 'w+') as f:
        #     json.dump(results, f, indent=4)

# with open('/hd2/wangzichuan/Evalmuse/EvalMuse-main/submission/2_output.json', 'w+') as f:
#     json.dump(results, f, indent=4)
