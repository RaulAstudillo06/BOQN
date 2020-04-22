#! /usr/bin/env python3

r"""
Network GP model.
"""

from __future__ import annotations
from typing import Any, List, Optional, Union
import torch
from botorch.models.model import Model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.posteriors import Posterior


class NetworkGP(Model):
    r"""
    """
    
    def __init__(self, dag, train_X, train_Y, train_Yvar=None, indices_X=None, outcome_transform=None) -> None:
        r"""
        """
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_Yvar = train_Yvar
        self.indices_X = indices_X
        self.outcome_transform = outcome_transform
        
        self.node_GPs = [None for k in range(self.n_nodes)]
        self.node_mlls = [None for k in range(self.n_nodes)]
        if self.indices_X is not None:
            for k in self.root_nodes:
                train_X_node_k = train_X[..., self.indices_X[k]]
                train_Y_node_k = train_Y[..., [k]]
                #print(k)
                #print(train_X_node_k)
                #print(train_Y_node_k)
                self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, outcome_transform=self.outcome_transform)
                self.node_mlls[k] = ExactMarginalLogLikelihood(self.node_GPs[k].likelihood, self.node_GPs[k])
                fit_gpytorch_model(self.node_mlls[k])
            
        for k in range(self.n_nodes):
            if self.node_GPs[k] is None:
                train_X_node_k = torch.cat([train_X, train_Y[..., self.dag.get_parent_nodes(k)]], 2)
                train_Y_node_k = train_Y[..., [k]]
                #print(k)
                #print(train_X_node_k)
                #print(train_Y_node_k)
                self.node_GPs.append(SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, outcome_transform=self.outcome_transform))
                self.node_GPs[k] = SingleTaskGP(train_X=train_X_node_k, train_Y=train_Y_node_k, outcome_transform=self.outcome_transform)
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
        return NetworkMultivariateNormal(self.node_GPs, self.dag, X, self.indices_X)
    
    def forward(self, x: Tensor) -> NetworkMultivariateNormal:
        return NetworkMultivariateNormal(self.node_GPs, self.dag, x, self.indices_X)
        
        
class NetworkMultivariateNormal(Posterior):
    def __init__(self, node_GPs, dag, X, indices_X=None):
        self.node_GPs = node_GPs
        self.dag = dag
        self.n_nodes = dag.get_n_nodes()
        self.root_nodes = dag.get_root_nodes()
        self.X = X
        self.indices_X = indices_X
        
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
        if len(sample_shape) > 1:
            print(error)
            
        nodes_samples = torch.zeros(sample_shape + self.event_shape)
        nodes_samples = nodes_samples.double()
        nodes_samples_available = [False for k in range(self.n_nodes)]
        for k in self.root_nodes:
            if self.indices_X is not None:
                X_node_k = self.X[..., self.indices_X[k]]
            else:
                X_node_k = self.X
            multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
            nodes_samples[..., k] = multivariate_normal_at_node_k.rsample(sample_shape, base_samples=base_samples[..., [k]])[..., 0]
            nodes_samples_available[k] = True
              
        while not all(nodes_samples_available):
            for k in range(self.n_nodes):
                parent_nodes = self.dag.get_parent_nodes(k)
                if not nodes_samples_available[k] and all([nodes_samples_available[j] for j in parent_nodes]):
                    for i in range(sample_shape[0]):
                        X_node_k = torch.cat([self.X, nodes_samples[i, :, :, parent_nodes]], 2)
                        multivariate_normal_at_node_k = self.node_GPs[k].posterior(X_node_k)
                        nodes_samples[i, :, :, k] = multivariate_normal_at_node_k.rsample(base_samples=base_samples[[1], :, :, [k]])[0, :, :, 0]
                    nodes_samples_available[k] = True
            
        return nodes_samples
        
                    
                    
                    
                    
                
        
        
    
    
        