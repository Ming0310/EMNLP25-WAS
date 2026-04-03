
sparsity=(0.4 0.6 0.75)
MODEL_NAME="/PATH/TO/Llama-2-7b-hf"
MODEL_TYPE="Llama-2-7B"
OUTPUT_PATH="./models/${MODEL_TYPE}"

for SPARSITY in "${sparsity[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python was/ppl_test.py \
    --model $MODEL_NAME \
    --was_path $OUTPUT_PATH \
    --sparsity $SPARSITY \
    --model_type $MODEL_TYPE \
    --eval_ppl \
    --greedy_flag \
    --batch_size auto:4.0 \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    --fewshot 0
done