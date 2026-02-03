layer=3
weight=6
mixing_ratio=0.15
draft_model=/home/tiantianyi/code/DuoDecoding/qwen/Qwen3-14B


WANDB_PROJECT=specdecpp python3 specdec_pp/train.py \
    --data_path /home/tiantianyi/code/DuoDecoding/src/SpecDec_pp/data/alpaca_data_qwen_14b_32b/train.json \
    --eval_data_path /home/tiantianyi/code/DuoDecoding/src/SpecDec_pp/data/alpaca_data_qwen_14b_32b/dev.json \
    --output_dir exp-qwen-weight${weight}-layer${layer}-mix${mixing_ratio} \
    --model_name_or_path ${draft_model} \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 8 \
    --logging_steps 5 \
    --eval_strategy epoch \
    --per_device_eval_batch_size 4 \
    --weight_mismatch ${weight} \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --resnet_num_layers ${layer} \
    --mixing_ratio ${mixing_ratio}
