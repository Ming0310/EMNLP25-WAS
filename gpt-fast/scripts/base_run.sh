model=llama2-7b
parent_path="/PATH/TO/CONVERTED_MODEL_DIR"

CUDA_VISIBLE_DEVICES=0 python generate.py --compile --checkpoint_path $parent_path/$model/model.pth --interactive