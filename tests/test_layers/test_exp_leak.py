import torch

from sinabs.layers import ExpLeak, ExpLeakSqueeze


def test_leaky_basic():
    time_steps = 100
    tau_mem = torch.tensor(30.0)
    input_current = torch.rand(time_steps, 2, 7, 7)
    layer = ExpLeak(tau_mem=tau_mem)
    membrane_output = layer(input_current)

    assert not layer.does_spike
    assert input_current.shape == membrane_output.shape
    assert torch.isnan(membrane_output).sum() == 0
    assert membrane_output.sum() > 0


def test_leaky_squeezed():
    batch_size = 10
    time_steps = 100
    tau_mem = torch.tensor(30.0)
    input_current = torch.rand(batch_size * time_steps, 2, 7, 7)
    layer = ExpLeakSqueeze(tau_mem=tau_mem, batch_size=batch_size)
    membrane_output = layer(input_current)

    assert input_current.shape == membrane_output.shape
    assert torch.isnan(membrane_output).sum() == 0
    assert membrane_output.sum() > 0


def test_leaky_norm_input():
    # - Make sure outputs are scaled according to normalization factor

    batch_size = 10
    time_steps = 100
    trailing_dims = (2, 7, 7)
    # Different tau_mem between 10 and 100
    tau_mem = (torch.rand(*trailing_dims) * 90) + 10
    input_current = torch.rand(batch_size, time_steps, *trailing_dims)
    layer = ExpLeak(tau_mem=tau_mem, norm_input=False)
    layer_norm = ExpLeak(tau_mem=tau_mem, norm_input=True)
    membrane_output = layer(input_current)
    membrane_output_norm = layer_norm(input_current)

    alpha_mem = torch.exp(-1.0 / tau_mem)
    normalization_factor = 1 - alpha_mem

    assert torch.allclose(membrane_output * normalization_factor, membrane_output_norm)

    # - Make sure sum under impulse response curve is independent of tau_mem when normalizing
    # Use more time steps to make sure impulse response has decayed for all neurons
    batch_size = 2
    time_steps = 1000
    layer_norm = ExpLeak(tau_mem=tau_mem, norm_input=True)
    input_current = torch.zeros((batch_size, time_steps, *trailing_dims))
    input_current[:, 0] = 1

    # Sum membrane output over time - should be 1 for all dimensions
    membrane_output_integrated = layer_norm(input_current).sum(1)

    assert torch.allclose(
        membrane_output_integrated,
        torch.ones_like(membrane_output_integrated),
        atol=5e-5,
    )


def test_leaky_membrane_decay():
    batch_size = 10
    time_steps = 100
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.zeros(batch_size, time_steps, 2, 7, 7)
    input_current[:, 0] = 1 / (1 - alpha)  # only inject current in the first time step
    layer = ExpLeak(tau_mem=tau_mem, norm_input=True)
    membrane_output = layer(input_current)

    # first time step is not decayed
    membrane_decay = alpha ** (time_steps - 1)

    # account for rounding errors with .isclose()
    assert torch.allclose(
        membrane_output[:, 0], torch.tensor(1.0)
    ), "Output for first time step is not correct."
    assert (
        membrane_output[:, -1] == layer.v_mem
    ).all(), "Output of last time step does not correspond to last layer state."
    assert torch.isclose(
        layer.v_mem, membrane_decay, atol=1e-08
    ).all(), "Neuron membrane potentials do not seems to decay correctly."


def test_leaky_v_mem_recordings():
    batch_size, time_steps = 10, 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = ExpLeak(tau_mem=30.0, record_states=True)
    membrane_output = layer(input_current)

    layer.recordings["v_mem"].sum().backward()
    assert layer.tau_mem.grad is not None

    assert layer.recordings["v_mem"].shape == membrane_output.shape
    assert "i_syn" not in layer.recordings.keys()
