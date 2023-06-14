from itertools import product
import pytest

periodic_exp_args = product((0.5, 1, 2), (0.5, 1, 2), (0.5, 1, 2))
@pytest.mark.parametrize(
    "grad_width,grad_scale,spike_threshold",
    periodic_exp_args
)
def test_periodic_exponential(grad_width, grad_scale, spike_threshold):
    from sinabs.activation import PeriodicExponential
    import torch

    grad_fn = PeriodicExponential(grad_width=grad_width, grad_scale=grad_scale)
    
    # Test with random values
    vmem_rand = torch.rand(100)
    surrogate_gradient = grad_fn(vmem_rand, spike_threshold)

    assert (surrogate_gradient >= 0).all()
    assert (surrogate_gradient <= grad_scale).all()

    # Test with multiples of threshold
    vmem_thr = torch.tensor([n * spike_threshold for n in range(1, 5)])
    surrogate_gradient_thr = grad_fn(vmem_thr, spike_threshold)
    assert torch.allclose(surrogate_gradient_thr, torch.tensor(grad_scale).float())

    # Test with values smaller than threshold - must be monotonous
    vmem_small = torch.linspace(-spike_threshold, spike_threshold, 50)
    surrogate_gradient_small = grad_fn(vmem_small, spike_threshold)
    assert (torch.diff(surrogate_gradient_small) > 0).all()
