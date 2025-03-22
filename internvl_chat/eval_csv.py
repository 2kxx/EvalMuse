import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import os



folder_path = "/hd2/wangzichuan/Evalmuse/EvalMuse-main/submission/eval_lora"
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        labels_a = []
        preds_a = []
        labels_e = []
        preds_e = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                row = line.strip().split(",") 
                labels_a.append(float(row[1]))
                preds_a.append(float(row[2]))

                if (len(row)-3)%2 == 0:
                    half = (len(row)-3)//2
                    e_l = row[3:half+3]
                    e_p = row[half+3:]
                    e_l = [float(x) for x in e_l]
                    e_p = [float(x) for x in e_p]
                    labels_e.extend(e_l)
                    preds_e.extend(e_p)
                else:
                    print(row)

        plcc, _ = pearsonr(labels_a, preds_a)
        srcc, _ = spearmanr(labels_a, preds_a)

        y_true_binary = (np.array(labels_e) >= 0.5).astype(int)
        y_pred_binary = (np.array(preds_e) >= 0.5).astype(int)
        accuracy = np.mean(np.array(y_true_binary) == np.array(y_pred_binary))
        
        final_score = (plcc + srcc) /4 + accuracy / 2

        print(f"For {filename}:")
        print(f"PLCC: {plcc:.4f}")
        print(f"SRCC: {srcc:.4f}")
        print(f"ACC: {accuracy:.4f}")
        print(f"Final score: {final_score:.4f}")
        print("\n")



# aggregated_data = {} 
# agg_num = [8,9,10,11,12]
# labels_a = []
# preds_a = []
# labels_e = []
# preds_e = []
# file_count = 0

# for filename in os.listdir(folder_path):
#     for i in agg_num:
#         if str(i) in filename:
#             file_path = os.path.join(folder_path, filename)
#             file_count += 1 

#             if file_count == 1:         
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     for line in f:
#                         row = line.strip().split(",") 
#                         labels_a.append(float(row[1]))
#                         preds_a.append(float(row[2]))
#                         if (len(row)-3)%2 == 0:
#                             half = (len(row)-3)//2
#                             e_l = row[3:half+3]
#                             e_p = row[half+3:]
#                             e_l = [float(x) for x in e_l]
#                             e_p = [float(x) for x in e_p]
#                             labels_e.extend(e_l)
#                             preds_e.extend(e_p)
#                         else:
#                             print(row)
#             else:
#                 labels_a_ = []
#                 preds_a_ = []
#                 labels_e_ = []
#                 preds_e_ = []
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     for line in f:
#                         row = line.strip().split(",") 
#                         labels_a_.append(float(row[1]))
#                         preds_a_.append(float(row[2]))
#                         if (len(row)-3)%2 == 0:
#                             half = (len(row)-3)//2
#                             e_l = row[3:half+3]
#                             e_p = row[half+3:]
#                             e_l = [float(x) for x in e_l]
#                             e_p = [float(x) for x in e_p]
#                             labels_e_.extend(e_l)
#                             preds_e_.extend(e_p)
#                         else:
#                             print(row)

#                     labels_a += labels_a_
#                     preds_a += preds_a_
#                     labels_e += labels_e_
#                     preds_e += preds_e_

# labels_a = [x / len(agg_num) for x in labels_a]
# preds_a = [x / len(agg_num) for x in preds_a]
# labels_e = [x / len(agg_num) for x in labels_e]
# preds_e = [x / len(agg_num) for x in preds_e]

# plcc, _ = pearsonr(labels_a, preds_a)
# srcc, _ = spearmanr(labels_a, preds_a)

# y_true_binary = (np.array(labels_e) >= 0.5).astype(int)
# y_pred_binary = (np.array(preds_e) >= 0.5).astype(int)
# accuracy = np.mean(y_true_binary == y_pred_binary)

# final_score = (plcc + srcc) / 4 + accuracy / 2

# print("Aggregated Results:")
# print(f"PLCC: {plcc:.4f}")
# print(f"SRCC: {srcc:.4f}")
# print(f"ACC: {accuracy:.4f}")
# print(f"Final score: {final_score:.4f}")
