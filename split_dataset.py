import json
import random

# 设置路径
input_file = '/hd2/tangzhenchen/project/EvalMuse-internvl/processed_train2.jsonl'
train_output_file = '/hd2/tangzhenchen/project/EvalMuse-internvl/processed_train_split.jsonl'
eval_output_file = '/hd2/tangzhenchen/project/EvalMuse-internvl/processed_eval_split.jsonl'

# 设置划分比例
train_ratio = 0.9  # 90% 训练，10% 验证

# 读取原始 JSONL 文件
with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 打乱顺序
random.shuffle(lines)

# 计算划分数量
total = len(lines)
train_count = int(train_ratio * total)

# 划分
train_lines = lines[:train_count]
eval_lines = lines[train_count:]

# 写入新的 train 文件
with open(train_output_file, 'w', encoding='utf-8') as f_train:
    f_train.writelines(train_lines)

# 写入新的 eval 文件
with open(eval_output_file, 'w', encoding='utf-8') as f_eval:
    f_eval.writelines(eval_lines)

print(f"划分完成：共 {total} 条数据 -> 训练集 {len(train_lines)} 条，验证集 {len(eval_lines)} 条")
