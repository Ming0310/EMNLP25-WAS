import os
import sys
import argparse
import torch
from utils.utils import get_sparse_model
import numpy as np
from functools import partial
import optuna
from eval_test.evaluate import eval_ppl
from eval_test.datautils import get_loaders
from model import LlamaSparseForCausalLM, LlamaSparseConfig, MistralSparseForCausalLM, MistralSparseConfig
from transformers import AutoConfig, AutoModelForCausalLM
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'utils'))
sys.path.append(os.path.join(parent_dir, 'eval_test'))
AutoConfig.register("llama_sparse", LlamaSparseConfig)
AutoModelForCausalLM.register(LlamaSparseConfig, LlamaSparseForCausalLM)
AutoConfig.register("mistral_sparse", MistralSparseConfig)
AutoModelForCausalLM.register(MistralSparseConfig, MistralSparseForCausalLM)

def object_func(trial, model, was_path, args):
    layers = model.model.layers
    sparsity_rates = []
    for i in range(len(layers)):
        if i < int(1):
            sparsity_rates.append(trial.suggest_float(f"sparsity_layer_{i}", args.target_sparsity - 0.02, args.target_sparsity))
        else:
            lower_bound = sparsity_rates[i - 1]
            upper_bound = max(args.target_sparsity + 0.02, lower_bound)
            upper_bound = min(upper_bound, lower_bound + 0.005)
            sparsity_rates.append(trial.suggest_float(f"sparsity_layer_{i}", lower_bound, upper_bound))
        
        
    sparsity_penalty = abs(np.mean(sparsity_rates) - args.target_sparsity) * 2500
    if sparsity_penalty > 25:
        return sparsity_penalty * 3
    model.load_greedy_sparsities(was_path+"/lookup", None, sparsity_rates)
    
    _, testloader = get_loaders(
        'wikitext2', seed=2, model=args.model_name, seqlen=2048
    )
    ppl = eval_ppl(model, testloader, dev=torch.device('cuda:0'))
    return ppl
    
def optuna_tpe(model, model_type, was_path, args):
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="minimize", sampler=sampler, storage=f"sqlite:///{model_type}-{args.target_sparsity}.db", study_name=model_type+f"{args.target_sparsity}", load_if_exists=True,)
    objective_function = partial(object_func, model=model, was_path=was_path, args=args)
    study.optimize(objective_function, n_trials=50)
    print("best_configuration:", study.best_params)
    print("best_loss:", study.best_value)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--was_path", type=str, required=True)
    parser.add_argument("--target_sparsity", type=float, default=0.8, help="Target effective sparsity")

    args = parser.parse_args()

    histogram_path = os.path.join(args.was_path, 'histograms')

    from utils.utils import get_model_class_name

    class_name = get_model_class_name(args.model_name)
    assert class_name in ['LlamaSparseForCausalLM', 'MistralSparseForCausalLM', 'LlamaForCausalLM', 'MistralForCausalLM'], f"Model {args.model_name} not supported"

    SparseModel = LlamaSparseForCausalLM if "Llama" in class_name else MistralSparseForCausalLM

    model = get_sparse_model(args.model_name, device='cpu', histogram_path=histogram_path)

    os.makedirs(os.path.join(args.was_path, 'lookup'), exist_ok=True)

    optuna_tpe(model, args.model_type, args.was_path, args)