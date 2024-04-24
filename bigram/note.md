# Bigram
_[pytorch manual](https://pytorch.org/docs/stable/index.html)_
## Goal
- train a bigram language model with the `names.txt` which contains a lot of names, and then generate some "new names".
## Usage of `Torch`
- key concept: tensor 
- API: `torch.tensor`, `torch.Tensor`, `torch.multinomial`, `torch.Generator`, `torch.sum`, **`broadcasting semantic`**
## Implementation
- probability and statistics `bigram.py`
  - performing counts and nomalizing those counts
  - space optimization: use `'.'` to express the meaning "start" and "end"
- neural network `bigram-nn.py`
  - `one_hot()` for encoding
  - _Q: Why should we feed float type instead of integer into neural netword?_
  - neural netword is more scalable and has an equivalent to smoothing
  - the use of strength of regularization (for loss function)
  - how to set loss function strongly based on the application scenario and your requirements (aka. what kind of nn is great in your view)
## Evaluation
- the concept to measure quality of bigram language model is **likeihood**
  - use _log(likelihood)_ instead of likelihood for large extent convenient
  - use _average negative log(likelihood)_ as loss function (the lower, the better)
    - the loss function should be made up only of differentiable operations for backpropergation in the neural network.
## Optimization
- **model smoothing**
