import torch
import math

class Langermann:
    
    def __init__(self):
        self.c = torch.tensor([1., 2., 5., 2., 3.])
        self.A = torch.tensor([[3., 5., 2., 1., 7.], [5., 2., 1., 4., 9.]])
        
        
    def evaluate(self, X):
        print(X)
        c = self.c
        A = self.A
        X_scaled = 10 * X
        output = torch.zeros(X_scaled.shape[:-1] + torch.Size([6]))
        for j in range(5):
            for i in range(2):
                output[..., j] += torch.pow(X_scaled[..., i] - A[i, j], 2)
        
        for j in range(5):
            output[..., 5] += c[j] * torch.exp(-output[..., j]/math.pi) * torch.cos(math.pi * output[..., j])
        print(output)   
        return output