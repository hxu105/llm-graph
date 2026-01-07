# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import json
import os
import sys
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from datasets import load_from_disk
from accelerate.utils import is_xpu_available
from torch_geometric.data import Data

from models import LlagaForCausalLM, LlagaConfig
from configs import quantization_config as QUANT_CONFIG

def main(
    model_name,
    peft_model: str=None,
    quantization: str = None, # Options: 4bit, 8bit
    device: str = 'auto',
    max_new_tokens =256, #The maximum numbers of tokens to generate
    min_new_tokens:int=0, #The minimum numbers of tokens to generate
    dataset_dir: str=None,
    seed: int=42, #seed value for reproducibility
    safety_score_threshold: float=0.5,
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=True, # Enable safety check woth Saleforce safety flan t5
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    enable_llamaguard_content_safety: bool = False,
    **kwargs
):
    dataset = load_from_disk(dataset_dir)['test']
    
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model_config = LlagaConfig.from_pretrained(model_name)
    
    quant_config = QUANT_CONFIG()
    if quantization in ['int4', 'int8']:
        bnb_config = quant_config.create_bnb_config(quantization)
    else:
        bnb_config = None
    
    if bnb_config:
        model = LlagaForCausalLM.from_pretrained(
            model_name, 
            config=model_config, 
            quantization_config=bnb_config, 
            device_map=device,
            **kwargs
        )
    else:
        dtype = quantization if quantization in ["auto", None] else getattr(torch, quantization)
        model = LlagaForCausalLM.from_pretrained(
            model_name, 
            config=model_config, 
            dtype=dtype, 
            device_map=device,
            **kwargs
        )
    # if peft_model:
    #     model = load_peft_model(model, peft_model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    num_evaluation = 0
    num_corrects = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataset))
        for idx, instance in pbar:
            chat = tokenizer.apply_chat_template(instance['conversation'])
            graph = Data(x=torch.FloatTensor(instance['x']).to(getattr(torch, quantization)), edge_index=torch.LongTensor(instance['edge_index']))
            tokens= torch.tensor(chat).long()
            tokens= tokens.unsqueeze(0)
            attention_mask = torch.ones_like(tokens)
            if is_xpu_available():
                tokens = tokens.to("xpu:0")
                graph = graph.to("xpu:0")
            else:
                tokens= tokens.to("cuda:0")
                graph = graph.to("cuda:0")
            outputs = model.generate(
                input_ids=tokens,
                attention_mask=attention_mask,
                graph=graph,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs
            )
            output_ids = outputs[0][tokens.size(-1):]
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            ground_truth = instance['label']
            # print("User input and model output deemed safe.")
            # print(f"Model output: {output_text}; Ground truth: {ground_truth}")
            # print("\n==================================\n")
            num_evaluation += 1
            num_corrects += 1 if output_text.strip() == ground_truth else 0
            pbar.set_description(f"Accuracy {num_corrects / num_evaluation}")
        



if __name__ == "__main__":
    fire.Fire(main)