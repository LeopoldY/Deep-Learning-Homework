import numpy as np

label = [1, 0, 0]

p = [0.6, 0.25, 0.15]

p_log = np.log(p)

l = -1 * np.dot(label, p_log.T)
print(l)