# f(x) = sin(x); 用数值微分法和自动求导的方式求导数

import torch
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

# 数值微分法
def numerical_diff(x):
    h = 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

x = np.arange(-5.0, 5.0, 0.1)
y = f(x)
dy_num = numerical_diff(x)

# 自动求导
x_torch = torch.tensor(x, requires_grad=True)
y_torch = torch.sin(x_torch)
y_torch.sum().backward()
dy_auto = x_torch.grad

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(8, 8))

ax1.plot(x, y, label="f(x)")
ax1.legend()

ax2.plot(x, dy_num, label="f'(x) (numerical)")
ax2.legend()

ax3.plot(x, dy_auto, label="f'(x) (automatic)")
ax3.legend()

plt.show() 

