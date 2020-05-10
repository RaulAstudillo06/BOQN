#!/usr/bin/env python3

import math
import torch
from gpytorch.kernels import Kernel
from gpytorch.kernels import MaternKernel
from botorch.models import SingleTaskGP
from gpytorch.priors.torch_priors import GammaPrior


class NodeMaternKernel(Kernel):
    r"""
    """

    has_lengthscale = True

    def __init__(self, main_input_indices, n_Y_indices, train_X_aux, train_Y_aux, nu=2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(NodeMaternKernel, self).__init__(**kwargs)
        self.main_input_indices = main_input_indices
        self.n_Y_indices = n_Y_indices
        self.nu = nu
        self.error_input_indices = [i for i in range(train_X_aux.shape[-1] - n_Y_indices) if i not in main_input_indices]
        #
        train_X_aux_y = train_X_aux[..., -n_Y_indices:]
        train_X_aux_main = torch.cat([train_X_aux[..., main_input_indices], train_X_aux_y], -1)
        aux_model = SingleTaskGP(train_X=train_X_aux_main, train_Y=train_Y_aux)
        batch_shape = aux_model._aug_batch_shape
        self.main_kernel = MaternKernel(nu=nu, ard_num_dims=train_X_aux_main.shape[-1], batch_shape=batch_shape, lengthscale_prior=GammaPrior(3.0, 6.0))
        #
        train_X_aux_error = train_X_aux[..., self.error_input_indices]
        aux_model = SingleTaskGP(train_X=train_X_aux_error, train_Y=train_Y_aux)
        batch_shape = aux_model._aug_batch_shape
        self.error_kernel = MaternKernel(nu=nu, ard_num_dims=train_X_aux_error.shape[-1], batch_shape=batch_shape, lengthscale_prior=GammaPrior(3.0, 6.0))

    def forward(self, x1, x2, diag=False, **params):
        x1_y = x1[..., -self.n_Y_indices:]
        x1_main = torch.cat([x1[..., self.main_input_indices], x1_y], -1)
        
        x2_y = x2[..., -self.n_Y_indices:]
        x2_main = torch.cat([x2[..., self.main_input_indices], x2_y], -1)
        
        x1_error = x1[..., self.error_input_indices]
        x2_error = x2[..., self.error_input_indices]
        #if x1.shape[-1] > 2:
            #print(x1)
            #print(x1_main)
            #print(x1_error)
            #print(error)
        #print(x1_main.shape)
        #print(x2_main.shape)
        #print(x1_error.shape)
        #print(x2_error.shape)
        
        
        return self.main_kernel.forward(x1_main, x2_main, diag, **params) + self.error_kernel.forward(x1_error, x2_error, diag, **params)