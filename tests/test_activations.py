def test_lif_membrane_reset():
    batch_size = 10
    time_steps = 100
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.ones(batch_size, time_steps, 2, 7, 7) * 10 / (1-alpha) # inject lots of current
    layer = LIF(tau_mem=tau_mem, membrane_reset=True)
    spike_output = layer(input_current)

    assert spike_output.max() == 1