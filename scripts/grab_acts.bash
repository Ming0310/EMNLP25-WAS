# Specify output path to store activations and histograms
MODEL_NAME="/PATH/TO/Llama-2-7b-hf"
OUTPUT_PATH="./models/${MODEL_TYPE}"

CUDA_VISIBLE_DEVICES=0 python was/grab_acts.py \
--model_name "$MODEL_NAME" \
--output_path "$OUTPUT_PATH"