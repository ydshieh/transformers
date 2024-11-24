# raise RuntimeError("Dynamo is not supported on Python 3.13+")

breakpoint()

#from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM


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

# `tokenizers` is not ok neither
# <frozen importlib._bootstrap>:488: RuntimeWarning: The global interpreter lock (GIL) has been enabled to load module 'sentencepiece._sentencepiece', which has not declared that it can run safely without the GIL. To override this behavior and keep the GIL disabled (at your own risk), run with PYTHON_GIL=0 or -Xgil=0.
# Segmentation fault (core dumped)

#tokenizer = AutoTokenizer.from_pretrained(ckpt)



# sentencepiece is not ok with py13 and GIL is reenabled
#from transformers import GemmaTokenizer
# tokenizer = GemmaTokenizer.from_pretrained(ckpt)


model = model.to(device)
transformers.generation.utils.my_model = model


sequence = "Hey what's the plan"

#inputs = tokenizer.encode(sequence, return_tensors='pt').to(device)

inputs = torch.tensor([[     2,   6750,   1212, 235303, 235256,    573,   1780]],
       device='cuda:0')


model.generation_config.temperature = 1.0
model.generation_config.top_p = 1.0


t0 = time.time()
out = model.generate(inputs, do_sample=False, max_new_tokens=500, cache_implementation="static")
#out = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
dt = time.time() - t0
print(f'dt: {dt}') 

t0 = time.time()
out = model.generate(inputs, do_sample=False, max_new_tokens=500, cache_implementation="static")
#out = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
dt = time.time() - t0
print(f'dt: {dt}') 
