import argparse
import os
import math
from dataclasses import dataclass, field
import tqdm.auto as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from datasets import load_dataset
import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from peft import (
    get_peft_model,
    LoraConfig,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
    TaskType,
)

from trl import SFTTrainer


@dataclass
class FinetuneArguments:
    dataset_path: str = field()
    model_path: str = field()


@dataclass
class PEFTArguments:
    peft_mode: str = field(default="lora")
    lora_rank: int = field(default=8)
    num_virtual_tokens: int = field(default=32)  # Used for prompt tuning, prefix tuning and p-tuning
    mapping_hidden_dim: int = field(default=1024)


def get_peft_config(peft_args: PEFTArguments):
    if peft_args.peft_mode == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=peft_args.lora_rank,
            lora_alpha=32, lora_dropout=0.1,
            bias="none",
            # target_modules=['qkv_proj']
            # target_modules=['q_proj', 'v_proj']
            # target_modules=['query_key_value']
        )
    elif peft_args.peft_mode == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
            prefix_projection=True,
        )
    elif peft_args.peft_mode == "ptuning":
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
            encoder_hidden_size=peft_args.mapping_hidden_dim,
        )
    elif peft_args.peft_mode == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=peft_args.num_virtual_tokens,
        )
    else:
        raise KeyError(peft_args.peft_mode)
    return peft_config


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


class ModifiedTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            attention_mask=torch.ones_like(inputs["input_ids"]),
            labels=inputs["input_ids"],  # HF model does the slicing for us
        ).loss


def data_collator(features: list) -> dict:
    return {
        "input_ids": torch.stack([
            torch.LongTensor(f["input_ids"])
            for f in features
        ])
    }


def save_tunable_parameters(model, path):
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)


def main():
    finetune_args, peft_args, training_args = HfArgumentParser((
        FinetuneArguments,
        PEFTArguments,
        TrainingArguments,
    )).parse_args_into_dataclasses()
    
    print("Setup Data")
    # dataset = datasets.load_from_disk(finetune_args.dataset_path)
    dataset_name = "/data/kiho/autocode/codegen/finetuning/llama-peft-tuner/code_segments_py150k_sanitized_20"
    dataset = load_dataset(dataset_name, split="train")
    print("Setup Model")
    # model = transformers.LlamaForCausalLM.from_pretrained(
    model = transformers.AutoModelForCausalLM.from_pretrained(
        finetune_args.model_path,
        load_in_8bit=True,
        device_map='auto',
        cache_dir='/data/kiho/autocode/codegen',
        trust_remote_code=True
    )
    
    ##
    for name, param in model.named_parameters():
	    print(name, param.shape)
    ##
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(finetune_args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.lm_head = CastOutputToFloat(model.embed_out)  # Codegen ==> model.lm_head / Pythia ==> model.embed_out 
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    print("Setup PEFT")
    peft_config = get_peft_config(peft_args=peft_args)
    model = get_peft_model(model, peft_config)

    print("Train")
    # trainer = ModifiedTrainer(
    #     model=model,
    #     train_dataset=dataset,
    #     args=training_args,
    #     data_collator=data_collator,
    # )
    
    trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    )
    
    # trainer.train()
    
    with torch.autocast("cuda"): 
        trainer.train()
    
    save_tunable_parameters(model, os.path.join(training_args.output_dir, "params.p"))


if __name__ == "__main__":
    main()
