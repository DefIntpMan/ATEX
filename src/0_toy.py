# An example of higher-order gradients computation

import torch
from torch.autograd import grad

def nth_derivative(f, wrt, n):

    for i in range(n):

        grads = grad(f, wrt, create_graph=True)[0]
        f = grads.sum()

    return grads

x = torch.arange(4., requires_grad=True).reshape(2, 2)
loss = (x ** 4).sum()

print(nth_derivative(f=loss, wrt=x, n=3))