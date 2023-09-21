import json
from textwrap import dedent

with open('repositories/stanford_alpaca/alpaca_data.json', 'r', encoding='utf-8') as f:
    entries = json.load(f)

formatted_entries = []

for entry in entries:
    formatted_entry = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{entry['instruction']}\n\n### Input:\n{entry['input']}\n\n### Response:\n{entry['output']}"

    formatted_entries.append({ "text": formatted_entry })
    
with open('alpaca_text.jsonl', 'w', encoding='utf-8') as f:
    for entry in formatted_entries:
        f.write(json.dumps(entry) + '\n')
