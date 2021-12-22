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


# def test_lif_zero_grad():
#     torch.autograd.set_detect_anomaly(False)
#     batch_size = 7
#     time_steps = 100
#     n_neurons = 20
#     threshold = 100
#     tau_mem = torch.tensor(30.)
#     alpha = torch.exp(-1/tau_mem)
#     sl = LIF(tau_mem=tau_mem, threshold=threshold)
#     conv = torch.nn.Conv1d(
#         in_channels=time_steps,
#         out_channels=time_steps,
#         kernel_size=3,
#         padding=1,
#         groups=time_steps,
#     )
#     sl_0 = LIF(tau_mem=tau_mem, threshold=threshold)
#     conv_0 = torch.nn.Conv1d(
#         in_channels=time_steps,
#         out_channels=time_steps,
#         kernel_size=3,
#         padding=1,
#         groups=time_steps,
#     )
#     model = torch.nn.Sequential(conv, sl)

#     # Copy of the original model, where zero_grad will already be applied at beginning
#     model_zg = torch.nn.Sequential(conv_0, sl_0)
#     model_zg[0].weight.data = model[0].weight.data.clone()
#     model_zg[0].bias.data = model[0].bias.data.clone()

#     optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
#     optimizer_zg = torch.optim.SGD(model_zg.parameters(), lr=0.001)

#     sl_0.zero_grad()

#     data0, data1, data2 = torch.rand((3, batch_size, time_steps, n_neurons)) / (1-alpha)
#     out0 = model(data0)
#     out0_zg = model_zg(data0)

#     loss = torch.nn.functional.mse_loss(out0, torch.ones_like(out0))
#     loss_zg = torch.nn.functional.mse_loss(out0_zg, torch.ones_like(out0_zg))
#     loss.backward()
#     loss_zg.backward()

#     grads = [p.grad.data.clone() for p in model.parameters()]
#     grads_zg = [p.grad.data.clone() for p in model_zg.parameters()]

#     for g, g0 in zip(grads, grads_zg):
#         assert torch.isclose(g, g0).all()

#     optimizer.step()
#     optimizer.zero_grad()

#     # Detach state gradients to avoid backpropagating through stored state.
#     sl.zero_grad()

#     out1 = model(data1)

#     loss = torch.nn.functional.mse_loss(out1, torch.ones_like(out1))
#     loss.backward()

#     # Make sure that without detaching there is a RuntimeError
#     with pytest.raises(RuntimeError):
#         out2 = model(data2)

#         loss = torch.nn.functional.mse_loss(out2, torch.ones_like(out2))
#         loss.backward()
