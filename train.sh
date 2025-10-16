layer=3
weight=6
draft_model=/home/tiantianyi/code/DuoDecoding/llama/llama-68m

WANDB_PROJECT=specdecpp python3 specdec_pp/train.py \
    --data_path /home/tiantianyi/code/DuoDecoding/src/SpecDec_pp/data/train.json \
    --eval_data_path /home/tiantianyi/code/DuoDecoding/src/SpecDec_pp/data/dev.json \
    --output_dir exp-weight${weight}-layer${layer} \
    --model_name_or_path ${draft_model} \
    --bf16 True \
    --per_device_train_batch_size 4 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 8 \
    --logging_steps 5 \
    --evaluation_strategy epoch \
    --per_device_eval_batch_size 4 \
    --weight_mismatch ${weight} \
    --save_strategy no \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --resnet_num_layers ${layer} \
    --mixing_ratio 0.15