#! /usr/bin/env python3

r"""
Network GP model.
"""

from __future__ import annotations
from typing import Any, List, Optional, Union
import torch
from botorch.models.model import Model
from botorch.models import SingleTaskGP, FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
from node_kernel import NodeMaternKernel
from botorch import fit_gpytorch_model
from botorch.posteriors import Posterior
from botorch.models.transforms import Standardize
from copy import deepcopy
import time


class NetworkGP(Model):
    r"""
    """
    
    def __init__(self, dag, train_X, train_Y, train_Yvar=None, active_input_indices=None, main_input_indices=None) -> None:
        r"""
        """
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
        self.active_input_indices = active_input_indices
        self.main_input_indices = main_input_indices
        self.main_input_indices_rescaled = [[] for k in range(self.n_nodes)]

        for k in range(self.n_nodes):
            for i in range(len(self.main_input_indices[k])):
                for j in range(len(self.active_input_indices[k])):
                    if self.main_input_indices[k][i] == self.active_input_indices[k][j]:
                        self.main_input_indices_rescaled[k].append(deepcopy(j))

        self.node_GPs = [None for k in range(self.n_nodes)]
        self.node_mlls = [None for k in range(self.n_nodes)]
        self.normalization_constant_lower = [[None for j in range(len(self.dag.get_parent_nodes(k)))] for k in range(self.n_nodes)]
        self.normalization_constant_upper = [[None for j in range(len(self.dag.get_parent_nodes(k)))] for k in range(self.n_nodes)]

        for k in self.root_nodes:
            if self.active_input_indices is not None:
                train_X_node_k = train_X[..., self.active_input_indices[k]]
            else:
                train_X_node_k = train_X
            train_Y_node_k = train_Y[..., [k]]
            #self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, outcome_transform=Standardize(m=1, batch_shape=torch.Size([1])))
            self.node_GPs[k] = FixedNoiseGP(train_X=train_X_node_k, train_Y=train_Y_node_k, train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-4, outcome_transform=Standardize(m=1, batch_shape=torch.Size([1])))
            self.node_mlls[k] = ExactMarginalLogLikelihood(self.node_GPs[k].likelihood, self.node_GPs[k])
            fit_gpytorch_model(self.node_mlls[k])
            
        for k in range(self.n_nodes):
            if self.node_GPs[k] is None:
                aux = train_Y[..., self.dag.get_parent_nodes(k)].clone()
                for j in range(len(self.dag.get_parent_nodes(k))):
                    self.normalization_constant_lower[k][j] = torch.min(aux[..., j])
                    self.normalization_constant_upper[k][j] = torch.max(aux[..., j])
                    aux[..., j] = (aux[..., j] - self.normalization_constant_lower[k][j])/(self.normalization_constant_upper[k][j] - self.normalization_constant_lower[k][j])
                train_X_node_k = torch.cat([train_X[..., self.active_input_indices[k]], aux], 2)
                train_Y_node_k = train_Y[..., [k]]
                aux_model =  FixedNoiseGP(train_X=train_X_node_k, train_Y=train_Y_node_k, train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-4, outcome_transform=Standardize(m=1, batch_shape=torch.Size([1])))  
                batch_shape = aux_model._aug_batch_shape
                #self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, outcome_transform=Standardize(m=1, batch_shape=torch.Size([1])))
                #self.node_GPs[k] = FixedNoiseGP(train_X=train_X_node_k, train_Y=train_Y_node_k, train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-4, outcome_transform=Standardize(m=1, batch_shape=torch.Size([1])))
                if len(self.main_input_indices[k]) < len(self.active_input_indices[k]):
                    covar_module_node_k = ScaleKernel(NodeMaternKernel(self.main_input_indices_rescaled[k], len(self.dag.get_parent_nodes(k)), train_X_node_k, train_Y_node_k, nu=2.5, ard_num_dims=train_X_node_k.shape[-1], batch_shape=batch_shape, lengthscale_prior=GammaPrior(3.0, 6.0)), batch_shape=batch_shape, outputscale_prior=GammaPrior(2.0, 0.15))
                    self.node_GPs[k] = FixedNoiseGP(train_X=train_X_node_k, train_Y=train_Y_node_k, train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-4, covar_module=covar_module_node_k, outcome_transform=Standardize(m=1, batch_shape=torch.Size([1])))
                else:
                    self.node_GPs[k] = FixedNoiseGP(train_X=train_X_node_k, train_Y=train_Y_node_k, train_Yvar=torch.ones(train_Y_node_k.shape) * 1e-4, outcome_transform=Standardize(m=1, batch_shape=torch.Size([1])))
                self.node_mlls[k] = ExactMarginalLogLikelihood(self.node_GPs[k].likelihood, self.node_GPs[k])
                fit_gpytorch_model(self.node_mlls[k])
                
    def posterior(self, X: Tensor) -> NetworkMultivariateNormal:
        r"""Computes the posterior over model outputs at the provided points.
        Args:
            X: A `(batch_shape) x q x d`-dim Tensor, where `d` is the dimension
                of the feature space and `q` is the number of points considered
                jointly.
            observation_noise: If True, add the observation noise from the
                likelihood to the posterior. If a Tensor, use it directly as the
                observation noise (must be of shape `(batch_shape) x q`).
        Returns:
            A `GPyTorchPosterior` object, representing a batch of `b` joint
            distributions over `q` points. Includes observation noise if
            specified.
        """
        return NetworkMultivariateNormal(self.node_GPs, self.dag, X, self.active_input_indices, self.normalization_constant_lower, self.normalization_constant_upper)
    
    def forward(self, x: Tensor) -> NetworkMultivariateNormal:
        return NetworkMultivariateNormal(self.node_GPs, self.dag, x, self.active_input_indices, self.normalization_constant)
        
        
class NetworkMultivariateNormal(Posterior):
    def __init__(self, node_GPs, dag, X, indices_X=None, normalization_constant_lower=None, normalization_constant_upper=None):
        self.node_GPs = node_GPs
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.X = X
        self.active_input_indices = indices_X
        self.normalization_constant_lower = normalization_constant_lower
        self.normalization_constant_upper = normalization_constant_upper
        
    @property
    def device(self) -> torch.device:
        r"""The torch device of the posterior."""
        return "cpu"

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the posterior."""
        return torch.double

    @property
    def event_shape(self) -> torch.Size:
        r"""The event shape (i.e. the shape of a single sample) of the posterior."""
        shape = list(self.X.shape)
        shape[-1] = self.n_nodes
        shape = torch.Size(shape)
        return shape
    
    def rsample(self, sample_shape=torch.Size(), base_samples=None):
        #t0 =  time.time()
        nodes_samples = torch.empty(sample_shape + self.event_shape)
        nodes_samples = nodes_samples.double()
        nodes_samples_available = [False for k in range(self.n_nodes)]
        for k in self.root_nodes:
            #t0 =  time.time()
            if self.active_input_indices is not None:
                X_node_k = self.X[..., self.active_input_indices[k]]
            else:
                X_node_k = self.X
            multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
            if base_samples is not None:
                nodes_samples[..., k] = multivariate_normal_at_node_k.rsample(sample_shape, base_samples=base_samples[..., [k]])[..., 0]
            else:
                nodes_samples[..., k] = multivariate_normal_at_node_k.rsample(sample_shape)[..., 0]
            nodes_samples_available[k] = True
            #t1 = time.time()
            #print('Part A of the code took: ' + str(t1 - t0))
  
        while not all(nodes_samples_available):
            for k in range(self.n_nodes): 
                parent_nodes = self.dag.get_parent_nodes(k)
                if not nodes_samples_available[k] and all([nodes_samples_available[j] for j in parent_nodes]):
                    #t0 =  time.time()
                    parent_nodes_samples_normalized = nodes_samples[..., parent_nodes].clone()
                    for j in range(len(parent_nodes)):
                        parent_nodes_samples_normalized[..., j] = (parent_nodes_samples_normalized[..., j] - self.normalization_constant_lower[k][j])/(self.normalization_constant_upper[k][j] - self.normalization_constant_lower[k][j])
                    X_node_k = self.X[..., self.active_input_indices[k]]
                    X_node_k = X_node_k.unsqueeze(0).repeat(sample_shape[0], 1, 1, 1)
                    X_node_k = torch.cat([X_node_k, parent_nodes_samples_normalized], 3)
                    multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
                    if base_samples is not None:
                        nodes_samples[..., k] = (multivariate_normal_at_node_k.mean + torch.einsum('abcd,a->abcd', torch.sqrt(multivariate_normal_at_node_k.variance), torch.flatten(base_samples[..., k])))[..., 0]
                    else:
                        nodes_samples[..., k] = multivariate_normal_at_node_k.rsample()[0, ..., 0]
                    nodes_samples_available[k] = True
                    #t1 = time.time()
                    #print('Part B of the code took: ' + str(t1 - t0))
        #t1 = time.time()
        #print('Taking this sample took: ' + str(t1 - t0))
        return nodes_samples