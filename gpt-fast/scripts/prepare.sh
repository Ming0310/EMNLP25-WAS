model=llama2-7b
parent_path="/PATH/TO/CONVERTED_MODEL_DIR"

python scripts/convert_hf_checkpoint.py --checkpoint_dir $parent_path/$model
