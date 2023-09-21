# Tune LLaMa-7B on Alpaca Dataset using PEFT / LORA

Based on @zphang's [minimal-llama](https://github.com/zphang/minimal-llama) scripts. 

References:
  - https://github.com/zphang/minimal-llama/#peft-fine-tuning-with-8-bit
  - https://github.com/tloen/alpaca-lora
  - https://github.com/tatsu-lab/stanford_alpaca

Prereqs

```
conda install cuda -c nvidia/label/cuda-11.7.1
conda install pytorch=1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
git clone https://github.com/tatsu-lab/stanford_alpaca repositories/stanford_alpaca
python convert_alpaca_to_text.py
python download_model.py decapoda-research/llama-7b-hf
```

If on Windows, add the DLL to bitsandbytes and patch according to this comment (https://github.com/TimDettmers/bitsandbytes/issues/30#issuecomment-1455993902):

```
cd "$(python -c 'import site; print(site.getsitepackages()[1])')/bitsandbytes"
curl -LOJ https://github.com/james-things/bitsandbytes-prebuilt-all_arch/raw/main/0.37.0/libbitsandbytes_cudaall.dll
```

If on linux, replace the backticks with backslashes in the following commands.

Tokenize the dataset:

```
python tokenize_dataset.py `
  --tokenizer_path models/llama-7b-hf `
  --jsonl_path alpaca_text.jsonl `
  --save_path alpaca_text_tokenized `
  --max_seq_length 512
```

Run the finetuning script:

```
python finetune_peft.py `
    --model_path models/llama-7b-hf `
    --dataset_path alpaca_text_tokenized `
    --peft_mode lora `
    --lora_rank 8 `
    --per_device_train_batch_size 2 `
    --gradient_accumulation_steps 1 `
    --max_steps 2500 `
    --learning_rate 2e-4 `
    --fp16 `
    --logging_steps 10 `
    --output_dir models/alpaca-llama-7b-peft
```

Change the steps to 2500 for full training. Play around with steps and batch size to see how it affects the results.

Generate:
```
python fgenerate.py
```

