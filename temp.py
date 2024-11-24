from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
device = "cuda"

import transformers

ckpt = "google/gemma-2b"
#ckpt = "ydshieh-gemma-2b"

breakpoint()

ckpt = "ydshieh-gemma-2b"
model = AutoModelForCausalLM.from_pretrained(ckpt)

breakpoint()

ckpt = "google/gemma-2b"
#tokenizer = AutoTokenizer.from_pretrained(ckpt)

# sentencepiece is not ok with py13 and GIL is reenabled
from transformers import GemmaTokenizer
# tokenizer = GemmaTokenizer.from_pretrained(ckpt)


model = model.to(device)
transformers.generation.utils.my_model = model


sequence = "Hey what's the plan"

inputs = tokenizer.encode(sequence, return_tensors='pt').to(device)
model.generation_config.temperature = 1.0
model.generation_config.top_p = 1.0


t0 = time.time()
out = model.generate(inputs, do_sample=False, max_new_tokens=500, cache_implementation="static")
out = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
dt = time.time() - t0
print(f'dt: {dt}') 

t0 = time.time()
out = model.generate(inputs, do_sample=False, max_new_tokens=500, cache_implementation="static")
out = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
dt = time.time() - t0
print(f'dt: {dt}') 
