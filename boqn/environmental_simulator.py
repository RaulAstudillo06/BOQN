import math
from typing import Tuple

import torch
from torch import Tensor


class EnvironmentalSimulator:
    r"""
    The environmental simulator test function from Astudillo & Frazier 2019.
    """

    def __init__(
        self,
        grid: Tuple[Tensor, Tensor],
    ) -> None:
        self.grid = grid
        self.grid_s, self.grid_t = grid[0].reshape(-1, 1), grid[1].reshape(-1)
        self.input_dim = 4
        self.bounds = [(7.0, 13.0), (0.02, 0.12), (0.01, 3.0), (30.01, 30.295)]
        self.m_0 = 10.0
        self.d_0 = 0.07 
        self.l_0 = 1.505
        self.tau_0 =30.1525
        self.optimizers = [(self.m_0, self.d_0, self.l_0, self.tau_0)]
        self.target_values = self.evaluate(
            torch.tensor([[self.m_0, self.d_0, self.l_0, self.tau_0]])
        )

    def evaluate(self, X: Tensor) -> Tensor:
        t_shape = X.shape[:-1] + (1, 1)
        m, d = X[..., 0].view(t_shape), X[..., 1].view(t_shape)
        l, tau = X[..., 2].view(t_shape), X[..., 3].view(t_shape)
        p_1 = m / (4 * math.pi * d * self.grid_t).sqrt()
        p_2 = torch.exp(- self.grid_s.pow(2) / (4 * d * self.grid_t))
        t_diff_clamped = (self.grid_t - tau).clamp_min(1e-10)  # avoid div by 0
        p_3 = (self.grid_t > tau) * m / torch.sqrt(4 * math.pi * d * t_diff_clamped)
        p_4 = torch.exp(- (self.grid_s - l).pow(2) / (4 * d * t_diff_clamped))
        result = p_1 * p_2 + p_3 * p_4
        return result.reshape(X.shape[:-1] + (-1,))
