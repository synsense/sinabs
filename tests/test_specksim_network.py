def util_create_all_ones_snn(spike_threshold: float = 1.0):
    import torch
    from torch import nn
    from sinabs.from_torch import from_model
    input_shape = (1, 8, 8)
    # initialize ann
    ann = nn.Sequential(
        # 8, 8
        nn.Conv2d(
            in_channels=1,
            out_channels=2,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False
        ),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        # 4, 4
        nn.Conv2d(
            in_channels=2,
            out_channels=4,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False
        ),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)), 
        # 2, 2
        nn.Flatten(),
        nn.Linear(in_features=16, out_features=4, bias=False)
        # 4 
    )
    # set weights to 1
    for layer in ann:
        if hasattr(layer, "weight"):
            with torch.no_grad(): 
                layer.weight.data = torch.ones_like(layer.weight.data)
    
    # convert and return snn
    return (
        from_model(
            ann, 
            input_shape=input_shape, 
            batch_size=1, 
            add_spiking_output=True,
            spike_threshold=spike_threshold
        ).spiking_model, 
        input_shape
    )
    
def test_specksim_network_conversion():
    from sinabs.backend.dynapcnn.specksim import from_sequential
    snn, input_shape = util_create_all_ones_snn()
    specksim_network = from_sequential(snn, input_shape=input_shape)
    assert specksim_network is not None

def test_add_monitor():
    from sinabs.backend.dynapcnn.specksim import from_sequential
    snn, input_shape = util_create_all_ones_snn()
    specksim_network = from_sequential(snn, input_shape=input_shape)
    specksim_network.add_monitor(1)
    assert len(specksim_network.monitors) == 1
    assert specksim_network.monitors[1]["graph"]
    assert specksim_network.monitors[1]["sink"]

def test_add_monitors():
    from sinabs.backend.dynapcnn.specksim import from_sequential
    snn, input_shape = util_create_all_ones_snn()
    specksim_network = from_sequential(snn, input_shape=input_shape)
    specksim_network.add_monitors([0, 1])
    assert len(specksim_network.monitors) == 2
    assert specksim_network.monitors[0]["graph"]
    assert specksim_network.monitors[0]["sink"]
    assert specksim_network.monitors[1]["graph"]
    assert specksim_network.monitors[1]["graph"]

def test_specksim_network_forward_pass():
    import numpy as np
    from sinabs.backend.dynapcnn.specksim import from_sequential
    snn, input_shape = util_create_all_ones_snn()
    specksim_network = from_sequential(snn, input_shape=input_shape)
    input_event = np.array([0,0,0,0], dtype=specksim_network.output_dtype)
    output_event = specksim_network(input_event)
    specksim_network.reset_states()
    assert len(output_event) > 0

def test_specksim_network_forward_pass_with_monitor():
    import numpy as np
    from sinabs.backend.dynapcnn.specksim import from_sequential
    snn, input_shape = util_create_all_ones_snn()
    specksim_network = from_sequential(snn, input_shape=input_shape)
    specksim_network.add_monitor(0)
    input_event = np.array([0,0,0,0], dtype=specksim_network.output_dtype)
    output_event = specksim_network(input_event)
    monitored_events = specksim_network.read_monitor(0)
    assert len(output_event) > 0
    assert len(monitored_events) > 0
