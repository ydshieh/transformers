import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"
ckpt = "google/gemma-2b"

model = AutoModelForCausalLM.from_pretrained(ckpt)
config = model.config
config.num_hidden_layers = 1
#config.vocab_size = 16
#config.intermediate_size = 16
#config.num_attention_heads = 2
#config.num_key_value_heads = 2
#config.head_dim = 16
#config.max_length = 16

model = type(model)(config=config)
model = model.to(device)
model.eval()


tokenizer = AutoTokenizer.from_pretrained(ckpt)

sequence = "Hey what's the plan" * 1
inputs = tokenizer.encode(sequence, return_tensors='pt').to(device)
inputs = torch.zeros_like(inputs, device=device)

#breakpoint()

N_WORKERS = 2
N_ITER = 10

#streams = [torch.cuda.Stream(device=device) for _  in range(N_WORKERS)]

x = torch.rand(size=(128*1, 128*1)).to(device)
w = torch.rand(size=(128*1, 128*1)).to(device)

def foo():
    s = torch.cuda.Stream(device=device)
    with torch.cuda.stream(s):
        o = 0
        with torch.no_grad():
            for i in range(N_ITER):
                #torch.cuda.nvtx.range_push('iter{}'.format(i))
                #out = torch.matmul(x, w)
                #o = o + out
                out = model(inputs)
                o = o + out.logits
            #torch.cuda.nvtx.range_pop()
        # print(o.device)
        #print(o.device)
        torch.cuda.synchronize()

import threading


import datetime

for i in range(20):
    s = datetime.datetime.now()

    for idx in range(N_WORKERS):
        t = threading.Thread(target=foo, args=())
        t.start()
        t.join()

    d = (datetime.datetime.now() - s).total_seconds()
    print(d)


exit(0)



import torch
import torch.nn as nn

import datetime
s = datetime.datetime.now()


N_WORKERS = 1


device = torch.device(0)

streams = [torch.cuda.Stream(device=device) for _ in range(N_WORKERS)]

x = torch.rand(size=(128*1, 128*1)).to(device)
w1 = torch.rand(size=(128*1, 128*1)).to(device)
    #w2 = torch.rand(size=(1024*4, 1024*4)).to(device)


def run(iters=1000):


    for i in range(iters):

        outputs = []

        torch.cuda.nvtx.range_push('iter{}'.format(i))

        for j in range(N_WORKERS):
            with torch.cuda.stream(streams[j]):
               o = 0
               out = x.matmul(w1)
               o = o + out
            outputs.append(o.device)

        torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize()
        #print([o for o in outputs])


if __name__=='__main__':
    # warmup

    #torch.cuda.cudart().cudaProfilerStart()

    for k in range(10):

        torch.cuda.synchronize()
        s = datetime.datetime.now()
        run()
        torch.cuda.synchronize()
        d = (datetime.datetime.now() - s).total_seconds()
        print(d)


    #torch.cuda.cudart().cudaProfilerStop()
