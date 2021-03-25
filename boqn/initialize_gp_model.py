import torch
from botorch.models import FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize


def initialize_gp_model(X, Y, Yvar=None):
    # define model
    print(X.shape)
    print(Y.shape)
    if Y.ndim == 1:
        Y = Y.unsqueeze(dim=-1)
    model = FixedNoiseGP(X, Y, torch.ones(Y.shape) * 1e-6, outcome_transform=Standardize(m=Y.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model