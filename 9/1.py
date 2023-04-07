import torch.nn as nn
import torch


class parallelModul(nn.Module):
    def __init__(self, net1, net2) -> None:
        super(parallelModul, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x):
        x1 = self.net1(x)
        x2 = self.net2(x)
        return torch.cat((x1, x2), dim=1)
    
net1 = nn.Sequential(nn.Flatten(),
                    nn.LazyLinear(64),
                    nn.ReLU(),
                    nn.Linear(64, 20))
net2 = nn.Sequential(nn.Flatten(),
                    nn.LazyLinear(64),
                    nn.ReLU(),
                    nn.Linear(64, 10))

net3 = nn.Sequential(nn.Flatten(),
                    nn.Linear(30, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10))

net = nn.Sequential(net1, 
                    parallelModul(net1, net2), 
                    net3)

X = torch.rand(2, 20)
y = net(X)

print(y)


