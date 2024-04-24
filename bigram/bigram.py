import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
N = torch.zeros((27, 27), dtype=torch.int32)
# convert the characters to integer
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0 
itos = {i+1:s for i, s in enumerate(chars)}
itos[0] = '.'

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

g = torch.Generator().manual_seed(114514)
new_names = set() 
P = N.float()
# `keepdim=True` here make sense
P /= P.sum(1, keepdim=True)
for _ in range(10):
    ix = 0
    out = []
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, generator=g).item()
        ch = itos[ix]
        if ch == '.':
            break
        out.append(ch)
    new_names.add(''.join(out)) 

# measure the bigram language model
logprob = 0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob += torch.log(prob)    
        n += 1
loss = (-logprob) / n
print(loss.item())
