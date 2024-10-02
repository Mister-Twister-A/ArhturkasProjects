# used for testing self-attention 



import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1941)
B,T,C = 4,8,32
x = torch.randn(B,T,C)
# head
num_heads = 16
key_layer = nn.Linear(C, num_heads, bias=False)
query_layer = nn.Linear(C, num_heads, bias=False)
value_layer = nn.Linear(C,num_heads,bias=False)
key = key_layer(x)
query = query_layer(x)
wei = query @ key.transpose(-2,-1) * num_heads**-0.5




tril = torch.tril(torch.ones(T,T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
value = value_layer(x)
c = wei @ value
print(wei)


