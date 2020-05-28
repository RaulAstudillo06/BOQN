import torch
import math

class Langermann5:
    
    def __init__(self, d=10, m=5):
        self.d = d
        self.m = m
        self.c = torch.tensor([0.806, 0.517, 1.5, 0.908, 0.9])
        self.A = torch.tensor([[9.681, 0.667, 4.783, 9.095, 3.517, 9.325, 6.544, 0.211, 5.122, 2.020],
                               [9.400, 2.041, 3.788, 7.931, 2.882, 2.672, 3.568, 1.284, 7.033, 7.374],
                               [8.025, 9.152, 5.114, 7.621, 4.564, 4.711, 2.996, 6.126, 0.734, 4.982],
                               [2.196, 0.415, 5.649, 6.979, 9.510, 9.166, 6.304, 6.054, 9.377, 1.426],
                               [8.074, 8.777, 3.467, 1.863, 6.708, 6.349, 4.534, 0.276, 7.633, 1.567]])
        self.A = torch.transpose(self.A, 0, 1)
        
    def evaluate(self, X):
        c = self.c
        A = self.A
        X_scaled = 10 * X
        output = torch.zeros(X_scaled.shape[:-1] + torch.Size([self.m + 1]))
        for j in range(self.m):
            for i in range(self.d):
                output[..., j] += torch.pow(X_scaled[..., i] - A[i, j], 2)
        
        for j in range(self.m):
            output[..., self.m] += c[j] * torch.exp(-output[..., j]/math.pi) * torch.cos(math.pi * output[..., j])
        return output