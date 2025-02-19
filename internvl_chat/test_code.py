from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig
from lmdeploy.vl import load_image
import torch.nn.functional as F
import torch
import os


# {'excellent':48920, 'good':15338, 'fair':59444, 'poor':299, 'bad':17222}
# {'yes':9583, 'no':2917}


model = '/hd2/wangzichuan/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/merged_internvl2_5_8b_dynamic_res_2nd_finetune_lora'
image_folder = '/hd2/tangzhenchen/dataset/EvalMuse/images'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))
gen_config = GenerationConfig(top_k=40, top_p=0.8, temperature=0.8, output_logits='generation', logprobs=5)

image = load_image('/hd2/tangzhenchen/dataset/EvalMuse/images/SDXL-Turbo/00110.png')
prompt = 'A puffin sitting in booth while eating a pastry at a diner. Etching'
q_a = f"This image is generated from the following prompt: {prompt}. How would you rate the alignment of this image with the prompt? Please respond with a single word."
sess = pipe.chat((q_a, image), gen_config=gen_config)
logits = sess.response.logits
probs = F.softmax(logits, dim=1)
topk_values, topk_indices = torch.topk(probs, k=5, dim=-1)
logprobs = sess.response.logprobs
print(sess.response.text)
print(topk_values, topk_indices)
print(logprobs)


data = {"id": "54", "width": 1024, "height": 1024, "image": "IF/00196.png", "conversations": [{"from": "human", "value": "<image>\nThis image is generated from the following prompt: 'Gaming carrot'. How would you rate the alignment of this image with the prompt? Please respond with a single word."}, {"from": "gpt", "value": "Bad."}, {"from": "human", "value": "Is 'carrot' present in the image?"}, {"from": "gpt", "value": "No."}, {"from": "human", "value": "Is the activity 'gaming' present in the image?"}, {"from": "gpt", "value": "No."}]}
image = load_image(os.path.join(image_folder, data['image']))
prompt = 'Gaming carrot'
q_a = f"This image is generated from the following prompt: {prompt}. How would you rate the alignment of this image with the prompt? Please respond with a single word."
sess = pipe.chat((q_a, image), gen_config=gen_config)
logits = sess.response.logits
probs = F.softmax(logits, dim=1)
topk_values, topk_indices = torch.topk(probs, k=5, dim=-1)
logprobs = sess.response.logprobs
print(sess.response.text)
print(topk_values, topk_indices)
print(logprobs)
print(probs[0][17222])
q_e = "Is 'carrot' present in the image?"
sess = pipe.chat(q_e, session=sess, gen_config=gen_config)
logits = sess.response.logits
probs = F.softmax(logits, dim=1)
topk_values, topk_indices = torch.topk(probs, k=5, dim=-1)
logprobs = sess.response.logprobs
print(sess.response.text)
print(topk_values, topk_indices)
print(logprobs)


data = {"id": "23", "width": 1024, "height": 1024, "image": "Kandinsky3/00840.png", "conversations": [{"from": "human", "value": "<image>\nThis image is generated from the following prompt: 'black and white alien hieroglyphs, very contrasted'. How would you rate the alignment of this image with the prompt? Please respond with a single word."}, {"from": "gpt", "value": "Poor."}, {"from": "human", "value": "Is 'alien hieroglyphs' present in the image?"}, {"from": "gpt", "value": "No."}, {"from": "human", "value": "Is the color 'black and white' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the attribute 'very contrasted' present in the image?"}, {"from": "gpt", "value": "Yes."}]}
image = load_image(os.path.join(image_folder, data['image']))
prompt = 'black and white alien hieroglyphs, very contrasted'
q_a = f"This image is generated from the following prompt: {prompt}. How would you rate the alignment of this image with the prompt? Please respond with a single word."
sess = pipe.chat((q_a, image), gen_config=gen_config)
logits = sess.response.logits
probs = F.softmax(logits, dim=1)
topk_values, topk_indices = torch.topk(probs, k=5, dim=-1)
logprobs = sess.response.logprobs
print(sess.response.text)
print(topk_values, topk_indices)
print(logprobs)


data = {"id": "136", "width": 1024, "height": 1024, "image": "SSD1B/00251.png", "conversations": [{"from": "human", "value": "<image>\nThis image is generated from the following prompt: 'Pink tree beside a large lake, landscape in the style of realism'. How would you rate the alignment of this image with the prompt? Please respond with a single word."}, {"from": "gpt", "value": "Excellent."}, {"from": "human", "value": "Is 'tree' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is 'lake' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is 'landscape' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is 'realism' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the color 'pink' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the attribute 'large' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the spatial relationship 'beside' present in the image?"}, {"from": "gpt", "value": "Yes."}]}
image = load_image(os.path.join(image_folder, data['image']))
prompt = 'Pink tree beside a large lake, landscape in the style of realism'
q_a = f"This image is generated from the following prompt: {prompt}. How would you rate the alignment of this image with the prompt? Please respond with a single word."
sess = pipe.chat((q_a, image), gen_config=gen_config)
logits = sess.response.logits
probs = F.softmax(logits, dim=1)
topk_values, topk_indices = torch.topk(probs, k=5, dim=-1)
logprobs = sess.response.logprobs
print(sess.response.text)
print(topk_values, topk_indices)
print(logprobs)
print(probs[0][48920])
q_e = "Is 'tree' present in the image?"
sess = pipe.chat(q_e, session=sess, gen_config=gen_config)
logits = sess.response.logits
probs = F.softmax(logits, dim=1)
topk_values, topk_indices = torch.topk(probs, k=5, dim=-1)
logprobs = sess.response.logprobs
print(sess.response.text)
print(topk_values, topk_indices)
print(logprobs)
q_e = "Is 'lake' present in the image?"
sess = pipe.chat(q_e, session=sess, gen_config=gen_config)
logits = sess.response.logits
probs = F.softmax(logits, dim=1)
topk_values, topk_indices = torch.topk(probs, k=5, dim=-1)
logprobs = sess.response.logprobs
print(sess.response.text)
print(topk_values, topk_indices)
print(logprobs)


data = {"id": "166", "width": 1024, "height": 1024, "image": "PixArt-alpha/00947.png", "conversations": [{"from": "human", "value": "<image>\nThis image is generated from the following prompt: 'close view of a delicious sweet and perfect snake cheesecake piece, snake pieces on top, snake slices on top, snake head, award winning, 4 k, beautiful'. How would you rate the alignment of this image with the prompt? Please respond with a single word."}, {"from": "gpt", "value": "Good."}, {"from": "human", "value": "Is the attribute 'view' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is 'cheesecake piece' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is 'snake pieces' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is 'snake slices' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is 'snake head' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the attribute 'award winning' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the attribute '4 k' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the attribute 'beautiful' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the attribute 'close' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the attribute 'delicious' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the attribute 'sweet' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the attribute 'perfect' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the spatial relationship 'on top' present in the image?"}, {"from": "gpt", "value": "Yes."}]}
image = load_image(os.path.join(image_folder, data['image']))
prompt = 'close view of a delicious sweet and perfect snake cheesecake piece, snake pieces on top, snake slices on top, snake head, award winning, 4 k, beautiful'
q_a = f"This image is generated from the following prompt: {prompt}. How would you rate the alignment of this image with the prompt? Please respond with a single word."
sess = pipe.chat((q_a, image), gen_config=gen_config)
logits = sess.response.logits
probs = F.softmax(logits, dim=1)
topk_values, topk_indices = torch.topk(probs, k=5, dim=-1)
logprobs = sess.response.logprobs
print(sess.response.text)
print(topk_values, topk_indices)
print(logprobs)
print(probs[0][59444])


{"id": "32717", "width": 1024, "height": 1024, "image": "Kolors/02958.png", "conversations": [{"from": "human", "value": "<image>\nThis image is generated from the following prompt: 'Royal Blue leather wallet and Green metal helicopter'. How would you rate the alignment of this image with the prompt? Please respond with a single word."}, {"from": "gpt", "value": "Good."}, {"from": "human", "value": "Is the color 'Royal Blue' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the material 'leather' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is 'Wallet' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the color 'Green' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is the material 'metal' present in the image?"}, {"from": "gpt", "value": "Yes."}, {"from": "human", "value": "Is 'Helicopter' present in the image?"}, {"from": "gpt", "value": "Yes."}]}
