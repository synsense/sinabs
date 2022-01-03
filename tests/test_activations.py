import pytest
import torch
from sinabs.activation import (
    ActivationFunction,
    ALIFActivationFunction,
    MembraneSubtract,
    MembraneReset,
    MultiSpike,
    SingleSpike,
)


@pytest.mark.parametrize(
    "act_fn, output, v_mem",
    [
        (
            ActivationFunction(spike_fn=SingleSpike, reset_fn=MembraneSubtract()),
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.5, 0.3]),
        ),
        (
            ActivationFunction(spike_fn=SingleSpike, reset_fn=MembraneReset()),
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 0.3]),
        ),
        (
            ActivationFunction(spike_fn=MultiSpike, reset_fn=MembraneSubtract()),
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.5, 0.3]),
        ),
        (
            ActivationFunction(spike_fn=MultiSpike, reset_fn=MembraneReset()),
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.0, 0.3]),
        ),
    ],
)
def test_activation_functions(act_fn, output, v_mem):
    state = {"v_mem": torch.tensor([2.5, 0.3], requires_grad=True)}

    spikes, new_state = act_fn(state)

    assert torch.allclose(spikes, output)
    assert torch.allclose(new_state["v_mem"], v_mem)

    loss = torch.nn.functional.mse_loss(spikes, torch.ones_like(spikes))
    loss.backward()
    
    assert not (state['v_mem'].grad == 0).all()
    assert not torch.isnan(state['v_mem'].grad).any()
    assert not torch.isinf(state['v_mem'].grad).any()


@pytest.mark.parametrize(
    "act_fn, output, v_mem",
    [
        (
            ALIFActivationFunction(spike_fn=SingleSpike, reset_fn=MembraneSubtract()),
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.4, 1.5]),
        ),
        (
            ALIFActivationFunction(spike_fn=SingleSpike, reset_fn=MembraneReset()),
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.5]),
        ),
        (
            ALIFActivationFunction(spike_fn=MultiSpike, reset_fn=MembraneSubtract()),
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.1, 1.5]),
        ),
        (
            ALIFActivationFunction(spike_fn=MultiSpike, reset_fn=MembraneReset()),
            torch.tensor([2.0, 0.0]),
            torch.tensor([0.0, 1.5]),
        ),
    ],
)
def test_alif_activation_functions(act_fn, output, v_mem):
    state = {
        "v_mem": torch.tensor([0.7, 1.5], requires_grad=True),
        "threshold": torch.tensor([0.3, 2.0]),
    }
    spikes, new_state = act_fn(state)

    assert torch.allclose(spikes, output)
    assert torch.allclose(new_state["v_mem"], v_mem)
    assert torch.allclose(new_state["threshold"], state["threshold"])

    loss = torch.nn.functional.mse_loss(spikes, torch.ones_like(spikes))
    loss.backward()

    assert not (state['v_mem'].grad == 0).all()
    assert not torch.isnan(state['v_mem'].grad).any()
    assert not torch.isinf(state['v_mem'].grad).any()
