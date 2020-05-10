import torch
#import botorch
from botorch.models import FixedNoiseGP
from botorch.models.transforms import Standardize
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.distributions.multivariate_normal import MultivariateNormal

class GPs_in_series:
    
    def __init__(self, n_nodes, seed):
        self.n_nodes = n_nodes
        self.seed = seed
        self.GPs = []
        self.normalization_constant_lower = []
        self.normalization_constant_upper = []
        n_points = 10
        covar_module = ScaleKernel(MaternKernel(nu=2.5))
        old_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        X = torch.rand(torch.Size([1, n_points, 1]))
        mean_X = torch.zeros(torch.Size([1, n_points]))
        cov_X =  covar_module(X)
        prior_X = MultivariateNormal(mean_X, cov_X, validate_args=True)
        
        Y = prior_X.rsample()
        Y = Y.view(torch.Size([1, n_points, 1]))
        self.normalization_constant_lower.append(0.)
        self.normalization_constant_upper.append(0.)
        self.GPs.append(FixedNoiseGP(train_X=X, train_Y=Y, train_Yvar=torch.ones(Y.shape) * 1e-4, covar_module=ScaleKernel(MaternKernel(nu=2.5)), outcome_transform=Standardize(m=1, batch_shape=torch.Size([1]))))
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2))
        for k in range(1, n_nodes):
            torch.manual_seed(seed + k)
            X = torch.rand(torch.Size([1, n_points, 2]))
            cov_X =  covar_module(X)
            prior_X = MultivariateNormal(mean_X, cov_X, validate_args=True)
            Y = prior_X.rsample()
            Y = Y.view(torch.Size([1, n_points, 1]))
            self.normalization_constant_lower.append(torch.min(Y))
            self.normalization_constant_upper.append(torch.max(Y))
            self.GPs.append(FixedNoiseGP(train_X=X, train_Y=Y, train_Yvar=torch.ones(Y.shape) * 1e-4, covar_module=ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=2)), outcome_transform=Standardize(m=1, batch_shape=torch.Size([1]))))
            
        torch.random.set_rng_state(old_state)
        
    def evaluate(self, X):
        X_copy = X.clone()
        input_shape = X_copy.shape
        output = torch.empty(input_shape[:-1] + torch.Size([self.n_nodes]))
        for i in range(input_shape[0]):
            for j in range(input_shape[1]):
                output[i, j, 0] = self.GPs[0].posterior(torch.reshape(X_copy[i, j, 0], [1, 1, 1])).mean.flatten().detach()
                for k in range(1, self.n_nodes):
                    Y_scaled = (output[i, j, k - 1] - self.normalization_constant_lower[k])/(self.normalization_constant_upper[k] - self.normalization_constant_lower[k])
                    X_input_k = torch.tensor([X_copy[i, j ,k], Y_scaled])
                    X_input_k = X_input_k.view([1, 1, 2])
                    output[i, j, k] = self.GPs[k].posterior(X_input_k).mean.flatten().detach()
        return output
            
    
    