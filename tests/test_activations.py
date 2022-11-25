import pytest
import torch

from sinabs.activation import (
    MaxSpike,
    MembraneReset,
    MembraneSubtract,
    MultiSpike,
    SingleExponential,
    SingleSpike,
)


@pytest.mark.parametrize(
    "spike_fn, reset_fn, output, v_mem",
    [
        (
            SingleSpike,
            MembraneSubtract(),
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.5, 0.3]),
        ),
        (
            SingleSpike,
            MembraneReset(),
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 0.3]),
        ),
        (
            MultiSpike,
            MembraneSubtract(),
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.5, 0.3]),
        ),
        (
            MultiSpike,
            MembraneReset(),
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.0, 0.3]),
        ),
    ],
)
def test_activation_functions(spike_fn, reset_fn, output, v_mem):
    state = {"v_mem": torch.tensor([2.5, 0.3], requires_grad=True)}

    spike_threshold = 1.0

    input_tensors = [state[name] for name in spike_fn.required_states]
    spikes = spike_fn.apply(*input_tensors, spike_threshold, SingleExponential())
    new_state = reset_fn(spikes, state, spike_threshold)

    assert torch.allclose(spikes, output)
    assert torch.allclose(new_state["v_mem"], v_mem)

    loss = torch.nn.functional.mse_loss(spikes, torch.ones_like(spikes))
    loss.backward()

    assert not (state["v_mem"].grad == 0).all()
    assert not torch.isnan(state["v_mem"].grad).any()
    assert not torch.isinf(state["v_mem"].grad).any()


def test_maxspike_activation():
    spike_threshold = 1.0
    max_num_spikes = 4
    spike_fn = MaxSpike(max_num_spikes)
    reset_fn = MembraneSubtract()

    v_mem = torch.tensor([6.5, 1.3], requires_grad=True)
    state = {"v_mem": v_mem}
    spikes = spike_fn.apply(v_mem, spike_threshold, SingleExponential())
    new_state = reset_fn(spikes, state, spike_threshold)

    assert torch.allclose(spikes, torch.tensor([4., 1.]))
    assert torch.allclose(new_state["v_mem"], torch.tensor([2.5, 0.3]))

    loss = torch.nn.functional.mse_loss(spikes, torch.ones_like(spikes))
    loss.backward()

    assert not (state["v_mem"].grad == 0).all()
    assert not torch.isnan(state["v_mem"].grad).any()
    assert not torch.isinf(state["v_mem"].grad).any()


def test_periodic_exponential():
    from sinabs.activation import PeriodicExponential

    grad_fn = PeriodicExponential(grad_width=0.1, grad_scale=1.0)
    x = torch.arange(-5.0, 10.5, 0.01)
    # Must have 10 peaks
    assert torch.sum(grad_fn(x, 1.0) == 1) == 10
