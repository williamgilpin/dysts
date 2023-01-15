#!/usr/bin/python

import torch

print("has gpu: ", torch.cuda.is_available(), flush=True)

n = torch.cuda.device_count()
print(f"{n} devices detected.", flush=True)
for i in range(n):
    print(torch.cuda.device(i), flush=True)
    print(torch.cuda.get_device_name(i), flush=True)
    
