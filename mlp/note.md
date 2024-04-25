# MLP (MultiLayer Perceptron)
## Embedding Matrix (Tensor)
- the index for the tensor is interesting
  - e.g. in the `mlp`, the shape of `C[X]` is [32, 3, 2], which express three vectors (characters) in a three dimentional space.
- the underlying storage of tensor, which is related to the view attributes of tensor. Reading the [blog](http://blog.ezyang.com/2019/05/pytorch-internals/) for more details.
## Loss Function
- `torch.nn.functional.cross_entropy` is much better than what we do in `bigram`  
  -  it's more efficient in both forward pass and backward pass
  -  it's more reliable and general for processing some number (e.g. `exp()` might generate `inf` which causing `nan` in probability)
## Optimization
- the usage of `minibatch` in practice: 
- how to determine or adjust the learning rate (step size): 
- the relationship between _parameters size_ & _training data size_ & _loss value_ & _the quality of model_:
- train dataset, dev/validation dataset, test dataset: 
- overfitting and underfitting:
