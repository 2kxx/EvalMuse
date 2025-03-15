import json
import random

with open('/hd2/tangzhenchen/dataset/EvalMuse/train.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


output_data = []

ANSWER_LIST = [
    "It is ",
    "The alignment score is ",
    "The score is ",
    "The answer is ",
    "",
]

SHORT_QUESTION_LIST0 = [
    "How well does the image align with the prompt? Please provide a quality score.",
    "Evaluate the image-text alignment for this prompt and give a score.",
    "What is the quality score for the alignment of this image with the prompt?",
    "Assess the coherence of the image and text related to this prompt. Output a score."
]


SHORT_QUESTION_LIST = [
    "How well does the image align with the description of '{class_name}' in the prompt? Please provide a quality score.",
    "Evaluate the image-text alignment for '{class_name}' and give a score.",
    "What is the quality score for the alignment of this image with the '{class_name}' in this prompt?",
    "Assess the coherence of the image and prompt related to '{class_name}'. Output a score."
]


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
        # new_item["conversations"].append({"from": "human", "value": f"<image>\nThis image is generated from the following prompt: '{item['prompt']}'. " + random.choice(SHORT_QUESTION_LIST0)})
        new_item["conversations"].append({"from": "human", "value": f"<image>\nThis image is generated from the following prompt: '{item['prompt']}'. Evaluate the image-text alignment for this prompt and give a score."})
            # "value": f"This image is generated from the following prompt: '{item['prompt']}'. How would you rate the alignment of this image with the prompt? Please respond with a single word.\n<|image|>"

        # new_item["conversations"].append({"from": "gpt", "value": random.choice(ANSWER_LIST) + "<score1>."})
        new_item["conversations"].append({"from": "gpt", "value": f"The degree of text-image alignment in this photo is {total_score_rating}, with an overall alignment score of <score1> <score2>."})

        for key, score in item["element_score"].items():
            last_left = key.rfind('(')
            last_right = key.rfind(')')
            category = key[last_left+1:last_right].lower()
            object = key[:last_left].rstrip()
            # choice = random.randint(0, 1)
            ans = ""
            # if choice == 1:
            #     ans = ans + get_element_score_answer(score) + " "

            # if '-' in category:
            #     category = category.split('-')[0]
            # if '/' in category:
            #     category = category.split('/')[0]
            #
            # if category in ['object', 'human', 'animal', 'food', 'location']:
            #     # question = f"Is '{object}' present in the image?\n<|image|>"
            #     question = f"Is the '{object}' present in the image?"
            #     ans += "<score2>"
            # elif category in ['activity', 'attribute', 'color', 'material', 'shape']:
            #     # question = f"Is the {category} '{object}' present in the image?\n<|image|>"
            #     question = f"Is the {category} '{object}' present in the image?"
            #     ans += "<score3>"
            # elif category == 'counting':
            #     # question = f"Is the quantity '{object}' present in the image?\n<|image|>"
            #     question = f"Is the quantity '{object}' present in the image?"
            #     ans += "<score3>"
            # elif category == 'spatial':
            #     # question = f"Is the spatial relationship '{object}' present in the image?\n<|image|>"
            #     question = f"Is the spatial relationship '{object}' present in the image?"
            #     ans += "<score3>"
            # elif category == 'other':
            #     # question = f"Is '{object}' present in the image?\n<|image|>"
            #     question = f"Is the '{object}' present in the image?"
            #     ans += "<score2>"
            # else:
            #     # question = f"Is the {category} concept '{object}' present in the image?\n<|image|>"
            #     question = f"Is the {category} concept '{object}' present in the image?"
            #     ans += "<score3>"

            # question = random.choice(SHORT_QUESTION_LIST).format(class_name=object)
            question = f"Assess the coherence of the image and prompt related to '{object}'. Output a score."
            ans += "<score3> <score4>"
            answer = get_element_score_answer(score)
            scaled_score = round(score * 6)
            s = score * 6
            if abs(scaled_score - s) < 0.00001:
                scores.append(scaled_score)
            else:
                scores.append(s)

            new_item["conversations"].append({"from": "human", "value": question})
            # new_item["conversations"].append({"from": "gpt", "value": random.choice(ANSWER_LIST) + f"{ans}."})
            new_item["conversations"].append({"from": "gpt", "value": f"The appearance of the '{object}' in this photo is described as {answer}, with a corresponding score of {ans}."})
        new_item["scores"] = scores
        output_file.write(json.dumps(new_item, ensure_ascii=False) + '\n')
    # output_data.append(new_item)

# with open('processed_train2.jsonl', 'w+', encoding='utf-8') as output_file:
#     json.dump(output_data, output_file, ensure_ascii=False, indent=4)
