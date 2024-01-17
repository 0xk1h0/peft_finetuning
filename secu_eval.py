import pandas as pd
import numpy as np
import torch
import transformers
from finetune_peft import get_peft_config, PEFTArguments
from peft import get_peft_model
import os
from tqdm import tqdm
import argparse

sec_eval_df = pd.read_csv('/data/kiho/autocode/LLMSecEval/Dataset/LLMSecEval-prompts.csv', index_col = 0)
python_df = sec_eval_df[sec_eval_df.Language == "Python"]
python_df.reset_index(inplace=True)

device = torch.device('cuda')
# peft_mode = "prefix"

parser = argparse.ArgumentParser()
parser.add_argument('--peft_mode', required=False, help='Finetuning methods')
args = parser.parse_args()
peft_mode = args.peft_mode

model_path = 'Salesforce/codegen25-7B-mono'
peft_path = f"/data/kiho/autocode/codegen/finetuning/llama-peft-tuner/models/codegen25_7b_lora_adapter.p"
torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, cache_dir='/data/kiho/autocode/codegen/finetuning/llama-peft-tuner/models/').to(device)
peft_config = get_peft_config(peft_args=PEFTArguments(peft_mode=peft_mode))
model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
temperature = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

for i in tqdm(range(len(python_df))):
    for temp in temperature:
        batch = tokenizer(f"""\"\"\"\n{python_df['NL Prompt'][i].replace('<language>', 'python')}\n\"\"\"\n""", return_tensors="pt").to(device)

        with torch.no_grad():
            if temp == 0:
                out = model.generate(
                    **batch,
                    pad_token_id=tokenizer.eos_token_id,
                    # input_ids=batch["input_ids"],
                    # attention_mask=torch.ones_like(batch["input_ids"]),
                    # max_length=128,
                    max_new_tokens=512,
                    do_sample=False
                )
            else:
                out = model.generate(
                **batch,
                pad_token_id=tokenizer.eos_token_id,
                # input_ids=batch["input_ids"],
                # attention_mask=torch.ones_like(batch["input_ids"]),
                # max_length=128,
                max_new_tokens=512,
                do_sample=True,
                temperature = temp,
                )
        
        with open(f"/data/kiho/autocode/codegen/finetuning/llama-peft-tuner/sec_{peft_mode}_25_7b/{python_df.iloc[i]['Prompt ID']}_temp_{temp}.py", 'w') as f:
        # with open(f"/data/kiho/autocode/codegen/finetuning/llama-peft-tuner/sec_lora_25_7b/{python_df.iloc[i]['Prompt ID']}_temp_{temp}.py", 'w') as f:
            # f.write(tokenizer.decode(out[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]).replace("<|endoftext|>", '').replace("<|python|>", ''))
            f.write(tokenizer.decode(out[0]).replace("<|endoftext|>", '').replace("<|python|>", ''))

            # f.write(tokenizer.decode(out[0]).replace("<|endoftext|>", ''))