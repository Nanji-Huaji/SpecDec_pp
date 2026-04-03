OUT=alpaca_data_gemma_270m_4b_it
FINAL=all.json
DATASET=tatsu-lab/alpaca

TARGET=/home/tiantianyi/code/DuoDecoding/google/gemma-3-4b-it
DRAFT=/home/tiantianyi/code/DuoDecoding/google/gemma-3-270m-it

mkdir -p ${OUT}
mkdir -p ${OUT}/tmp

TMP1=${OUT}/tmp/tmp1.json
TMP2=${OUT}/tmp/tmp2.json
TMP3=${OUT}/tmp/tmp3.json
TMP4=${OUT}/tmp/tmp4.json

python3 gen_dataset.py --dataset_name ${DATASET} --model_name ${TARGET} --mode hf --do_sample --output_file $TMP1 
python3 gen_assistant.py --model_name ${DRAFT} --do_sample --input_file $TMP1 --output_file $TMP2 
python3 gen_log_p.py --model_name ${DRAFT} --input_file $TMP2 --output_file $TMP3 
python3 gen_log_p.py --model_name ${TARGET} --input_file $TMP3 --output_file $TMP4 
python3 gen_acceptance.py --target_name ${TARGET} --draft_name ${DRAFT} --input_file $TMP4 --output_file ${OUT}/${FINAL}
