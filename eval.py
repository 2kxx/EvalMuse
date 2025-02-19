import time

import torch
import json
import csv
from transformers import BertTokenizer
from tqdm import tqdm
from lavis.models import load_model_and_preprocess, load_model
import os
from PIL import Image
from utils import compute_metrics, load_data
from torch.utils.data import DataLoader
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side='right')
tokenizer.add_special_tokens({"bos_token": "[DEC]"})

def binarize(lst):
    # 将列表中的元素按 0.5 阈值进行二值化
    return [1 if x >= 0.5 else 0 for x in lst]

def calculate_accuracy(true_list, pred_list):
    # 先二值化两个列表
    # breakpoint()
    true_bin = binarize(true_list)
    pred_bin = binarize(pred_list)

    # 计算相同元素的数量
    correct = sum([1 for t, p in zip(true_bin, pred_bin) if t == p])
    loss = torch.mean(torch.abs(torch.tensor(true_list)- torch.tensor(pred_list)))
    # 计算准确率
    accuracy = correct / len(true_list)
    return accuracy, loss

def is_sublist(lst1, lst2):
    return str(lst1)[1:-1] in str(lst2)[1:-1]

def get_index(list1,list2):
    len_list1 = len(list1)
    len_list2 = len(list2)
    for i in range(len_list2 - len_list1 + 1):
        if list2[i:i + len_list1] == list1:
            return i
    return 0

def is_sublist(lst1, lst2):
    return str(lst1)[1:-1] in str(lst2)[1:-1]

def custom_collate_fn(batch):
    """
    处理 DataLoader 传入的 batch，确保所有字段格式正确。
    """
    collated_batch = {
        'prompt': [item['prompt'] for item in batch],
        'img_path': [item['img_path'] for item in batch],
        'elements': [list(item['elements']) for item in batch]  # 解决 dict_keys 不能被默认 collate 的问题
    }
    return collated_batch

def eval(args):
    data = load_data(args.data_file, 'json')
    model, vis_processors, text_processors = load_model_and_preprocess("fga_blip2", "coco", device=device, is_eval=True)
    model.load_checkpoint(args.model_path)
    model.eval()

    batch_size = 128  # 确保 batch_size 适中
    times = []
    result_list = []

    # 预处理数据
    dataset = []
    for item in data:
        dataset.append({
            'prompt': item['prompt'],
            'img_path': os.path.join(args.dataset_dir, item['img_path']),
            'elements': item['element_score'].keys()  # 这里原本是 dict_keys，需要转 list
        })

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    for batch in tqdm(dataloader):
        start_time = time.time()
        images = [Image.open(img_path).convert("RGB") for img_path in batch['img_path']]
        images = torch.stack([vis_processors["eval"](img) for img in images]).to(device)

        prompts = batch['prompt']
        prompts = [text_processors["eval"](prompt) for prompt in prompts]
        prompts_ids = tokenizer(prompts).input_ids

        torch.cuda.empty_cache()
        with torch.no_grad():
            alignment_scores, scores = model.element_score(images, prompts)

        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

        for i in range(len(prompts)):
            elements_score = {}
            prompt_ids = prompts_ids[i]

            for element in batch['elements'][i]:  # 现在 batch['elements'][i] 是 list 而不是 dict_keys
                element_ = element.rpartition('(')[0]
                element_ids = tokenizer(element_).input_ids[1:-1]
                idx = get_index(element_ids, prompt_ids)

                if idx:
                    mask = torch.zeros_like(torch.tensor(prompt_ids))
                    mask[idx:idx + len(element_ids)] = 1
                    mask = torch.tensor(mask).to(device)
                    trimmed_len = min(scores[i].size(0), mask.size(0))

                    elements_score[element] = ((scores[i][:trimmed_len] * mask[:trimmed_len]).sum() / mask.sum()).item()
                else:
                    elements_score[element] = 0

            new_item = {
                'prompt': prompts[i],
                'img_path': batch['img_path'][i],
                'total_score': alignment_scores[i].item(),
                'element_score': elements_score
            }
            result_list.append(new_item)

    with open(args.save_path, 'w', newline='', encoding='utf-8') as file:
        json.dump(result_list, file, ensure_ascii=False, indent=4)

    avg_runtime = sum(times) / len(times) if times else 0
    use_cpu = 0
    use_extra_data = 0
    use_llm = 0
    other_description = "Nothing."

    # 写入 readme.txt
    with open("results/readme.txt", "w") as f:
        f.write(f"runtime per image [s] : {avg_runtime:.2f}\n")
        f.write(f"CPU[1] / GPU[0] : {use_cpu}\n")
        f.write(f"Extra Data [1] / No Extra Data [0] : {use_extra_data}\n")
        f.write(f"LLM[1] / Non-LLM[0] : {use_llm}\n")
        f.write(f"Other description : {other_description}\n")
    print("readme.txt has been generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='/hd2/tangzhenchen/dataset/EvalMuse/test.json')
    parser.add_argument('--save_path', type=str, default='results/output4.json')
    parser.add_argument('--model_path', type=str, default='/hd2/tangzhenchen/project/EvalMuse/lavis/output/FGA-BLIP2/20250215231/checkpoint_6.pth')
    parser.add_argument('--dataset_dir', type=str, default='/hd2/tangzhenchen/dataset/EvalMuse/images/')
    args = parser.parse_args()
    eval(args)



