model=llama2-7b
parent_path="/PATH/TO/CONVERTED_MODEL_DIR"
activation_path="/PATH/TO/ACTIVATION_DIR"

CUDA_VISIBLE_DEVICES=0 python generate.py --compile --checkpoint_path $parent_path/$model/model.pth \
--hist_path $activation_path/histograms --sparsity 0.4 --interactive \
--lookup_dir $activation_path/lookup