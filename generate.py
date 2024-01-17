import torch
import transformers
from finetune_peft import get_peft_config, PEFTArguments
from peft import get_peft_model

model_path = 'Salesforce/codegen2-1b'
# model_path = '/data/kiho/autocode/codegen/models--Salesforce--codegen2-1b'
# peft_path = 'models/codegen25_7b/checkpoint'
# peft_path = '0xk1h0/codegen25-7b-py150k-r20'
peft_path = '/data/kiho/autocode/codegen/finetuning/llama-peft-tuner/models/codegen2-1b-IA3/params.p'

torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, cache_dir='/data/kiho/autocode/codegen')
peft_config = get_peft_config(peft_args=PEFTArguments(peft_mode="IA3"))
model = get_peft_model(model, peft_config)
model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
batch = tokenizer("""
Generate python function to AES MODE encrypt.
""", return_tensors="pt")

with torch.no_grad():
    out = model.generate(
        **batch,
        # input_ids=batch["input_ids"],
        # attention_mask=torch.ones_like(batch["input_ids"]),
        # max_length=512,
        max_new_tokens=256,
        do_sample=True,
        temperature = 0.2,
        top_p=1
    )
print(tokenizer.decode(out[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"]))