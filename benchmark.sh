layer=3  
weight=6 
thres=0.3 

ckpt=exp-weight${weight}-layer${layer}

target_model=/home/tiantianyi/code/DuoDecoding/llama/tiny-llama-1.1b
draft_model=/home/tiantianyi/code/DuoDecoding/llama/llama-68m
data=/home/tiantianyi/code/DuoDecoding/src/SpecDec_pp/data/dev.json
SAVEPATH=/home/tiantianyi/code/DuoDecoding/src/SpecDec_pp/exp-weight6-layer3

python3 specdec_pp/evaluate.py \
  --model_name ${target_model} \
  --assistant_name ${draft_model} \
  --num_assistant_tokens_schedule ada \
  --data_path ${data} \
  --assist_acc_head_dir $ckpt\
  --do_sample \
  --random_seed 42 \
  --save_path ${SAVEPATH} \
  --stop_threshold ${thres} \
  --bound 2 20