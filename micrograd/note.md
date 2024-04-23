# MicroGrad 

## Key Concepts
- derivative
- gradient
- the **chain rule** in calculus
  - _dz/dx = (dz/dy) * (dy/dx)_

## `class Value`
- how to express the neural network (aka.mathematic expression)?
  - `data` & `children` & `op`
  - **syntax tree**: how to parse the expression 

## Backpropagation
- it's just a recursive application of the chain rule backwards through the computation graph

## Neural Network
- it can be regarded as neurons in human brain
- a mathematical expression (function)
- paras
  - weight`*`: the strength of connection between two neurons
  - bias`+`: the difficuly to trigger the neuron
- activation function: `tanh` or `sigmiod` ...
  - how to evaluate a activation function?

## procedure (implementation)
### `numpy`
- **topological sort**: it's a little trick to implement `backward()`
### `torch`
