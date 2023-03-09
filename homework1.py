# f(x) = sin(x); 用数值微分法和自动求导的方式求导数

import torch
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

def numerical_diff(x):
    h = 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

x = np.arange(-5.0, 5.0, 0.1)
y = f(x)
dy_num = numerical_diff(x)

x_torch = torch.tensor(x, requires_grad=True)
y_torch = torch.sin(x_torch)
y_torch.sum().backward()
dy_auto = x_torch.grad

plt.plot(x, y, label="f(x)")
plt.plot(x, dy_num, label="f'(x) (numerical)")
plt.plot(x, dy_auto, label="f'(x) (automatic)")
plt.legend()
plt.show()

