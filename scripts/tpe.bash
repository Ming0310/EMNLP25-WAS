MODEL_NAME="/PATH/TO/Llama-2-7b-hf"
MODEL_TYPE="Llama-2-7B"
OUTPUT_PATH="./models/${MODEL_TYPE}"

CUDA_VISIBLE_DEVICES=0 python was/tpe.py --model_name $MODEL_NAME --model_type $MODEL_TYPE --was_path $OUTPUT_PATH --target_sparsity 0.6