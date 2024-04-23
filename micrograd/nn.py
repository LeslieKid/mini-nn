import math
import numpy as np
import random

class Value:
    ''' stores a single scalar value and its gradient '''
    
    def __init__(self, data, _children=(), op='') -> None:
        self.data = data 
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = _children
        self.op = op
    
    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        data = self.data + other.data 
        out = Value(data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad             
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        data = self.data * other.data 
        out = Value(data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.grad * out.grad
        out._backward = _backward
        
        return out
    
    # implement the tanh function for more computational details (fine grained)
    def tanh(self):
        x = self.data
        data = (math.exp(2*x)-1) / (math.exp(2*x)+1)
        out = Value(data, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - data**2) * out.grad 
        out._backward = _backward 
         
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child) 
                topo.append(v)

        build_topo(self)
        for value in reversed(topo):
            value._backward()

class Neuron:
    ''' stores a Value and related weights and bias '''
    
    def __init__(self, nin) -> None:
        self.weights = [Value(random.uniform(-1, 1)) for _ in nin]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # weight * x + bias
        act = Value(sum(wi*xi for wi, xi in zip(self.weights, x)) + self.bias)
        out = act.tanh()
        return out 

class Layer:
    '''a set of neurons'''

    def __init__(self, nin, nout) -> None:
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs

class MLP:
    '''the multilayer perceptron'''

    def __init__(self, nin, nouts) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) 
        return x

def loss(pred, gt):
    # a function to measure the nn
    return sum((p-t) ** 2 for p, t in zip(pred, gt))
    