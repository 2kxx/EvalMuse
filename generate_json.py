import json


with open('/hd2/tangzhenchen/dataset/EvalMuse/train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

def get_total_score_rating(total_score):
    if 1 <= total_score < 1.5:
        return "bad"
    elif 1.5 <= total_score < 2.5:
        return "poor"
    elif 2.5 <= total_score < 3.5:
        return "fair"
    elif 3.5 <= total_score < 4.5:
        return "good"
    elif 4.5 <= total_score <= 5:
        return "excellent"
    return "unknown"


def get_element_score_answer(score):
    if score < 0.5:
        return "no"
    else:
        return "yes"

output_data = []

with open('processed_train2.jsonl', 'w', encoding='utf-8') as output_file:
    for idx, item in enumerate(data, start=1):
        model = item["img_path"].split('/')[0]
        if model in ['SD_v1.5', 'SDXL-Turbo', 'SD_v2.1', 'SD_v1.2']:
            width = 512
            height = 512
        else:
            width = 1024
            height = 1024
        # new_item = {
        #     "id": str(idx),
        #     "width": width,
        #     "height": height,
        #     "image": item["img_path"],
        #     "conversations": []
        # }
        new_item = {"id": str(idx), "width": width, "height": height, "image": item["img_path"], "conversations": [], "scores": []}
        scores = []
        total_score_rating = get_total_score_rating(item["total_score"])
        scores.append(item["total_score"])
        new_item["conversations"].append({"from": "human", "value": f"<image>\nThis image is generated from the following prompt: '{item['prompt']}'. How would you rate the alignment of this image with the prompt? Please respond with a single word."})
            # "value": f"This image is generated from the following prompt: '{item['prompt']}'. How would you rate the alignment of this image with the prompt? Please respond with a single word.\n<|image|>"

        new_item["conversations"].append({"from": "gpt", "value": total_score_rating.capitalize() + "."})

        for key, score in item["element_score"].items():
            last_left = key.rfind('(')
            last_right = key.rfind(')')
            category = key[last_left+1:last_right].lower()
            object = key[:last_left].rstrip()
            if '-' in category:
                category = category.split('-')[0]
            if '/' in category:
                category = category.split('/')[0]

            if category in ['object', 'human', 'animal', 'food', 'location']:
                # question = f"Is '{object}' present in the image?\n<|image|>"
                question = f"Is '{object}' present in the image?"
            elif category in ['activity', 'attribute', 'color', 'material', 'shape']:
                # question = f"Is the {category} '{object}' present in the image?\n<|image|>"
                question = f"Is the {category} '{object}' present in the image?"
            elif category == 'counting':
                # question = f"Is the quantity '{object}' present in the image?\n<|image|>"
                question = f"Is the quantity '{object}' present in the image?"
            elif category == 'spatial':
                # question = f"Is the spatial relationship '{object}' present in the image?\n<|image|>"
                question = f"Is the spatial relationship '{object}' present in the image?"
            elif category == 'other':
                # question = f"Is '{object}' present in the image?\n<|image|>"
                question = f"Is '{object}' present in the image?"
            else:
                # question = f"Is the {category} concept '{object}' present in the image?\n<|image|>"
                question = f"Is the {category} concept '{object}' present in the image?"
                print(category, object)

            answer = get_element_score_answer(score)
            scores.append(score)
            # new_item["conversations"].append(
            #     {
            #         "from": "human",
            #         "value": question
            #     }
            # )
            # new_item["conversations"].append(
            #     {
            #         "from": "gpt",
            #         "value": answer.capitalize() + "."
            #     }
            # )
            new_item["conversations"].append({"from": "human", "value": question})
            new_item["conversations"].append({"from": "gpt", "value": answer.capitalize() + "."})
        new_item["scores"] = scores
        output_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')
    # output_data.append(new_item)

# with open('processed_train2.jsonl', 'w+', encoding='utf-8') as output_file:
#     json.dump(output_data, output_file, ensure_ascii=False, indent=4)
