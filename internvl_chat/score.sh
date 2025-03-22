#!/bin/bash

MODEL_PATHS=(
    "/hd2/wangzichuan/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/evalmuse/merged_13_lora_64-internvl2_5_8b_dynamic_res_2nd_finetune_lora/checkpoint-3200"
    "/hd2/wangzichuan/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/evalmuse/merged_13_lora_64-internvl2_5_8b_dynamic_res_2nd_finetune_lora/checkpoint-4000"
    "/hd2/wangzichuan/InternVL/internvl_chat/work_dirs/internvl_chat_v2_5/evalmuse/merged_13_lora_64-internvl2_5_8b_dynamic_res_2nd_finetune_lora/checkpoint-4800"
)

SAVE_PATHS=(
    "/hd2/wangzichuan/Evalmuse/dataset/alignment/test/submission/lora64_3200.json"
    "/hd2/wangzichuan/Evalmuse/dataset/alignment/test/submission/lora64_4000.json"
    "/hd2/wangzichuan/Evalmuse/dataset/alignment/test/submission/lora64_4800.json"
)

for i in "${!MODEL_PATHS[@]}"; do
    MODEL="${MODEL_PATHS[$i]}"
    SAVE="${SAVE_PATHS[$i]}"
    
    echo "Running evaluation for model: $MODEL, save path: $SAVE"
    python score.py --model "$MODEL" --save_path "$SAVE"
done
