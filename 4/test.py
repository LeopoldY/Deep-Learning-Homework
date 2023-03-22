import numpy as  np
import torch

m1 = np.array([[0.1, 0.1, 0.2],
               [0.2, 0.1, 0.2]]) # model

m2 = np.array([[0.2, 0.3]]) # sample

ji = np.dot(m1.T, m2.T)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

p = softmax(torch.tensor(ji.T))

l = -1 * torch.log(p[0])

print(l)