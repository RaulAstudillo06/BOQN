import torch
from torch import Tensor
from botorch.optim import optimize_acqf
from custom_acqf_optimizer import custom_optimize_acqf


def optimize_acqf_and_get_suggested_point(
    acq_func,
    bounds,
    batch_size,
    posterior_mean=None,
    ) -> Tensor:
    """Optimizes the acquisition function, and returns a new candidate."""
    input_dim = bounds.shape[1]
    
    if posterior_mean is not None:
        baseline_candidate, _ = optimize_acqf(
            acq_function=posterior_mean,
            bounds=bounds,
            q=batch_size,
            num_restarts=10*input_dim,
            raw_samples=100*input_dim,
            options={"batch_limit": 5},
        )
    
        baseline_candidate = baseline_candidate.detach().view(torch.Size([1, batch_size, input_dim]))
    else:
        baseline_candidate = None
    
    candidate, acq_value = custom_optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=10*input_dim,
        raw_samples=100*input_dim,
        baseline_initial_conditions=baseline_candidate,
        options={"batch_limit": 5},
        #options={'disp': True, 'iprint': 101},
    )
    if baseline_candidate is not None:
        baseline_acq_value = acq_func.forward(baseline_candidate)[0].detach()
        print('Test begins')
        print(acq_value)
        print(baseline_acq_value)
        print('Test ends')
        if baseline_acq_value >= acq_value:
            print('Baseline candidate was best found.')
            candidate = baseline_candidate
            
    new_x = candidate.detach().view([batch_size, input_dim])
    return new_x