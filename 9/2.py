import torch
import torch.nn as nn
import torch.nn.functional as F

def makeLinkedNet(num_instances, module_class, *args, **kwargs):
    """
    该函数首先通过对每个实例调用 module_class(*args, **kwargs) 来创建一个包含 num_instances 个网络实例的列表。
    然后，它从实例列表创建一个 nn.Sequential 模块，并将其作为最终输出返回。

    参数：\n
    num_instances：要创建的网络实例数量\n
    module_class：要创建实例的网络类\n
    *args 和 **kwargs：要传递给网络构造函数的可选参数
    """
    instances = [module_class(*args, **kwargs) for _ in range(num_instances)]
    parallel_net = nn.Sequential(*instances)
    return parallel_net

# Define the network architecture
class MyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Generate random data to test the network
input_size = 5
hidden_size = 10
output_size = 5

x = torch.randn(1, input_size)
print(x)

# Create a parallel network with 4 instances of the MyNetwork class
num_instances = 4
parallel_net = makeLinkedNet(num_instances, MyNetwork, input_size=input_size, hidden_size=hidden_size, output_size=output_size)

# Test the parallel network with the random data
y_pred = parallel_net(x)


print(f"Predicted output:\n{y_pred}")

