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
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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
    loss = torch.mean(torch.abs(torch.tensor(true_list) - torch.tensor(pred_list)))
    # 计算准确率
    accuracy = correct / len(true_list)
    return accuracy, loss


def is_sublist(lst1, lst2):
    return str(lst1)[1:-1] in str(lst2)[1:-1]


def get_index(list1, list2):
    len_list1 = len(list1)
    len_list2 = len(list2)
    for i in range(len_list2 - len_list1 + 1):
        if list2[i:i + len_list1] == list1:
            return i
    return 0


def is_sublist(lst1, lst2):
    return str(lst1)[1:-1] in str(lst2)[1:-1]


def eval(args):
    data = load_data(args.data_file, 'json')
    data_new = []
    model, vis_processors, text_processors = load_model_and_preprocess("fga_blip2", "coco", device=device, is_eval=True)
    model.load_checkpoint(args.model_path)
    model.eval()

    times = []
    result_list = []
    for item in tqdm(data):
        start_time = time.time()
        elements = item['element_score'].keys()
        prompt = item['prompt']

        image = os.path.join(args.dataset_dir, item['img_path'])

        image = Image.open(image).convert("RGB")
        image = vis_processors["eval"](image).to(device)
        prompt = text_processors["eval"](prompt)
        prompt_ids = tokenizer(prompt).input_ids
        # breakpoint()

        torch.cuda.empty_cache()
        with torch.no_grad():
            alignment_score, scores = model.element_score(image.unsqueeze(0), [prompt])

        elapsed_time = time.time() - start_time
        times.append(elapsed_time)

        elements_score = dict()
        for element in elements:
            element_ = element.rpartition('(')[0]
            element_ids = tokenizer(element_).input_ids[1:-1]
            # breakpoint()

            idx = get_index(element_ids, prompt_ids)
            # breakpoint()
            if idx:
                mask = [0] * len(prompt_ids)
                mask[idx:idx + len(element_ids)] = [1] * len(element_ids)

                mask = torch.tensor(mask).to(device)
                elements_score[element] = ((scores * mask).sum() / mask.sum()).item()
            else:
                elements_score[element] = 0
        item['score_result'] = alignment_score.item()
        item['element_result'] = elements_score

        new_item = dict(prompt=item['prompt'], img_path=item['img_path'], total_score=item['score_result'],
                        element_score=item['element_result'])
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
    parser.add_argument('--save_path', type=str, default='results/output3.json')
    parser.add_argument('--model_path', type=str,
                        default='/hd2/tangzhenchen/project/EvalMuse/lavis/output/FGA-BLIP2/20250218225/checkpoint_14.pth')
    parser.add_argument('--dataset_dir', type=str, default='/hd2/tangzhenchen/dataset/EvalMuse/images/')
    args = parser.parse_args()
    eval(args)



