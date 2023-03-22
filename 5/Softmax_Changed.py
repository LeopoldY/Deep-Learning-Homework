import torch
import numpy as np

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def softmax_1(X):
    X_exp = torch.exp(X - max(X))
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def log_Softmax(X):
    return np.log(softmax_1(X))

m2 = np.array([[0.2, 0.3]])
print(log_Softmax(torch.Tensor(m2)))