import torch
import torch.nn.functional as F
import random

words = open('names.txt', 'r').read().splitlines()
random.shuffle(words)
chars = sorted(list(set(''.join(words))))
stoi = { s:i+1 for i, s in enumerate(chars) }
stoi['.'] = 0
itos = { i+1:s for i, s in enumerate(chars) }
itos[0] = '.'

block_size = 3
batch_size = 32

# build the dataset (X is input & Y is lable)
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch] 
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# seperate the dataset into train, dev/validation, test
n1 = int(len(words) * 0.8)
n2 = int(len(words) * 0.9)
Xtr, Ytr = build_dataset(words=words[:n1])
Xva, Yva = build_dataset(words=words[n1:n2])
 
g = torch.Generator().manual_seed(2147483647) # why should we need generator?
# embedding matrix
C = torch.randn((27, 10), generator=g, requires_grad=True)
W1 = torch.randn((30, 200), generator=g, requires_grad=True)
b1 = torch.randn(200, generator=g, requires_grad=True)
W2 = torch.randn((200, 27), generator=g, requires_grad=True)
b2 = torch.randn(27, generator=g, requires_grad=True)
parameters = [C, W1, b1, W2, b2]
model_size = sum(para.nelement() for para in parameters)
print(f'model size is {model_size}')

train_times = 1000000
for i in range(train_times):
    # minibatch construct 
    ix = torch.randint(0, Xtr.shape[0], (batch_size,))
    emb = C[Xtr[ix]]  

    # forward pass 
    h = torch.tanh(emb.view((batch_size, 30)) @ W1 + b1) 
    logits = h @ W2 + b2

    # loss function (average negative log likelihood)
    loss = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    for para in parameters:
        para.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < train_times * 0.6 else 0.01
    for para in parameters:
        para.data += -lr * para.grad

# calculate the loss for training data
emb = C[Xtr]
h = torch.tanh(emb.view((Xtr.shape[0], 30)) @ W1 + b1) 
logits = h @ W2 + b2
train_loss = F.cross_entropy(logits, Ytr) 
print(train_loss.item()) # 2.0237

# calculate the loss for validation data
emb = C[Xva]
h = torch.tanh(emb.view((Xva.shape[0], 30)) @ W1 + b1) 
logits = h @ W2 + b2
validation_loss = F.cross_entropy(logits, Yva)
print(validation_loss.item()) # 2.1215

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
      emb = C[torch.tensor([context])] 
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))
