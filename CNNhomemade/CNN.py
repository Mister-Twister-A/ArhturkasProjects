import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas 
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


mnist = pandas.read_csv("train.csv")

pictures = mnist.values.tolist()

#print(pictures[0])
pixels = np.array(pictures[3][1:], dtype=np.uint8)
pixels = pixels.reshape(28,28)
image = Image.fromarray(pixels)
image.show()


kernel = torch.tril(torch.ones(3,3))
kernel_no = torch.ones(3,3)
kernel_edge = torch.tensor([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])
kernel_gaussian = (1/16) * torch.tensor([
    [1,2,1],
    [2,4,2],
    [1,2,1]
])
print("kernel =", kernel)


class CNN():
    def __init__(self, kernel_, pixels_):
        self.kernel = kernel_
        self.pixels = torch.tensor(pixels_)
    
    def conv(self, pooling_type="sum"):
        size  = self.kernel.shape
        size_i = size[0]
        size_j = size[1]
        if size_i > self.pixels.shape[0] or size_j > self.pixels.shape[1]:
            print("Error: you are stupid kernel is bigger than photo lol")
            return "error"
        
        num_operations_i = self.pixels.shape[0] - size_i
        num_operations_j = self.pixels.shape[1] - size_j

        result = torch.zeros(num_operations_i,num_operations_j)

        for i in range(self.pixels.shape[0] - size_i):
            for j in range(self.pixels.shape[1] - size_j):
                batch = self.pixels[i: i + size_i, j : j + size_j]
                convoluted = F.relu(batch * self.kernel)
                pool_result = 0
                if pooling_type == "sum":
                    pool_result = torch.sum(convoluted)
                if pooling_type == "max":
                    pool_result = torch.max(convoluted)
                if pooling_type == "avg":
                    pool_result = torch.sum(convoluted) / (size_i * size_j)
                
                result[i,j] = pool_result
        return result

                
                    



cnn = CNN(kernel_gaussian, pixels)
res_pixels = cnn.conv(pooling_type="sum")



res_pixels = res_pixels.to(torch.int32)
res_pixels =  res_pixels.numpy()

conv_image = Image.fromarray(res_pixels)
conv_image.show()



print("lol")

