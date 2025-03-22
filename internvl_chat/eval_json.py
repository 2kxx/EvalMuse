import json
import numpy as np
from scipy.stats import spearmanr, pearsonr


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 计算 SRCC、PLCC 和 ACC
def evaluate_metrics(pred_json, label_json):
    total_scores_pred = []
    total_scores_label = []
    element_scores_pred = []
    element_scores_label = []

    for pred, label in zip(pred_json, label_json):
        # 计算 total_score 的 SRCC 和 PLCC
        total_scores_pred.append(pred["total_score"])
        total_scores_label.append(label["total_score"])

        # 计算 element_score 的 ACC
        pred_scores = pred["element_score"]
        label_scores = label["element_score"]

        # 确保 key 顺序一致
        common_keys = pred_scores.keys() & label_scores.keys()
        pred_values = [pred_scores[k] for k in common_keys]
        label_values = [label_scores[k] for k in common_keys]

        element_scores_pred.extend(pred_values)
        element_scores_label.extend(label_values)

    # 计算 SRCC 和 PLCC
    srcc = spearmanr(total_scores_pred, total_scores_label)[0]
    plcc = pearsonr(total_scores_pred, total_scores_label)[0]

    # 计算 ACC
    y_true_binary = (np.array(element_scores_label) >= 0.5).astype(int)
    y_pred_binary = (np.array(element_scores_pred) >= 0.5).astype(int)
    acc = np.mean(np.array(y_true_binary) == np.array(y_pred_binary))
    
    final_score = (plcc + srcc) /4 + acc / 2

    return srcc, plcc, acc, final_score

# 文件路径
pred_json_path = "/hd2/wangzichuan/IPCE/results/L14-norm-kl/merged_results.json"  # 预测 JSON
label_json_path = "/hd2/wangzichuan/Evalmuse/dataset/alignment/eval.json"  # 标签 JSON

# 读取数据
pred_json = load_json(pred_json_path)
label_json = load_json(label_json_path)

# 计算指标
srcc, plcc, acc, final_score = evaluate_metrics(pred_json, label_json)

# 输出结果
print(f"PLCC: {plcc:.4f}")
print(f"SRCC: {srcc:.4f}")
print(f"ACC: {acc:.4f}")
print(f"Final Score: {final_score:.4f}")
