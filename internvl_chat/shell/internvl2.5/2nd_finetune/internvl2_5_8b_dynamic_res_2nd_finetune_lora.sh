set -x

GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))


export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34230
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch
export CUDA_VISIBLE_DEVICES=1,2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


OUTPUT_DIR='work_dirs/internvl_chat_v2_5/internvl2_5_8b_dynamic_res_2nd_finetune_lora2'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 2
# batch size per gpu: 4
# gradient accumulation steps: 2
# total batch size: 16
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "/hd2/tangzhenchen/project/EvalMuse-internvl/internvl_chat/work_dirs/internvl_chat_v2_5/internvl2_5_8b_dynamic_res_2nd_finetune_lora_coco_merge_final" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/hd2/tangzhenchen/project/EvalMuse-internvl/internvl_chat/shell/data/evalmuse.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp True \
  --freeze_backbone True \
  --unfreeze_lm_head False\
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 4 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_steps 400 \
  --evaluation_strategy "steps" \
  --eval_steps 5 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
