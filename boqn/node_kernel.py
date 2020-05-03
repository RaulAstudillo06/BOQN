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

    def __init__(self, active_X_indices, n_Y_indices, train_X_aux, train_Y_aux, nu=2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(NodeMaternKernel, self).__init__(**kwargs)
        self.active_X_indices = active_X_indices
        self.n_Y_indices = n_Y_indices
        self.nu = nu
        #
        train_X_aux_active = train_X_aux[..., self.active_X_indices]
        train_X_aux_y = train_X_aux[..., -self.n_Y_indices:]
        train_X_aux_main = torch.cat([train_X_aux_active, train_X_aux_y], -1)
        aux_model = SingleTaskGP(train_X=train_X_aux_main, train_Y=train_Y_aux)
        batch_shape = aux_model._aug_batch_shape
        self.main_kernel = MaternKernel(nu=nu, ard_num_dims=train_X_aux_main.shape[-1], batch_shape=batch_shape, lengthscale_prior=GammaPrior(3.0, 6.0))
        #
        train_X_aux_error = train_X_aux[..., :-self.n_Y_indices]
        aux_model = SingleTaskGP(train_X=train_X_aux_error, train_Y=train_Y_aux)
        batch_shape = aux_model._aug_batch_shape
        self.error_kernel = MaternKernel(nu=nu, ard_num_dims=train_X_aux_error.shape[-1], batch_shape=batch_shape, lengthscale_prior=GammaPrior(3.0, 6.0))

    def forward(self, x1, x2, diag=False, **params):
        x1_active = x1[..., self.active_X_indices]
        x1_y = x1[..., -self.n_Y_indices:]
        x1_main = torch.cat([x1_active, x1_y], -1)
        
        x2_active = x2[..., self.active_X_indices]
        x2_y = x2[..., -self.n_Y_indices:]
        x2_main = torch.cat([x2_active, x2_y], -1)
        
        x1_error = x1[..., :-self.n_Y_indices]
        x2_error = x2[..., :-self.n_Y_indices]
        #print(x1_main.shape)
        #print(x2_main.shape)
        #print(x1_error.shape)
        #print(x2_error.shape)
        
        
        return self.main_kernel.forward(x1_main, x2_main, diag, **params) + self.error_kernel.forward(x1_error, x2_error, diag, **params)