import numpy as np
import matplotlib.pyplot as plt

def prelu(x, a):
    return np.maximum(0, x) + a * np.minimum(0, x)

x = np.linspace(-10, 10, 100)
y = prelu(x, 0.25)

plt.plot(x, y)
plt.title('pReLU Function')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
