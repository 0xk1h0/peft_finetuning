import torch
import transformers
from finetune_peft import get_peft_config, PEFTArguments
from peft import get_peft_model

model_path = 'Salesforce/codegen25-7b-mono'
# peft_path = 'models/codegen25_7b/checkpoint'
peft_path = '0xk1h0/codegen25-7b-py150k-r20'
# peft_path = 'models/alpaca-llama-7b-peft/params.p'

torch.set_default_tensor_type(torch.cuda.HalfTensor)
model = transformers.AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, cache_dir='models')
peft_config = get_peft_config(peft_args=PEFTArguments(peft_mode="lora"))
model = get_peft_model(model, peft_config)
# model.load_state_dict(torch.load(peft_path), strict=False)
torch.set_default_tensor_type(torch.cuda.FloatTensor)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
batch = tokenizer("""
### Generate AES MODE encrypt function.
""", return_tensors="pt")

with torch.no_grad():
    out = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=torch.ones_like(batch["input_ids"]),
        max_length=256,
        do_sample=True,
        temperature = 0.4,
        top_p=0.95
        
    )
print(tokenizer.decode(out[0]))