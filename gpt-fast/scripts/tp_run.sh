model=llama2-7b
parent_path="/PATH/TO/CONVERTED_MODEL_DIR"
activation_path="/PATH/TO/ACTIVATION_DIR"
time torchrun --standalone --nproc_per_node=4 generate.py --compile --checkpoint_path $parent_path/$model/model.pth --lookup_dir $activation_path/lookup --hist_path $activation_path/histograms --sparsity 0.5 --prompt "Hello, my name is "