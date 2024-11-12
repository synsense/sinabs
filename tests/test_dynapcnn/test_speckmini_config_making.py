import torch.nn as nn

import sinabs.layers as sl
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.from_torch import from_model

# Create a model which uses all kernel memories
ann_for_kernel_mem_test = nn.Sequential(
    *[
        # kernel memory = 16Ki
        nn.Conv2d(1, 256, (8, 8), (8, 8), bias=False),
        nn.ReLU(),
        # kernel memory = 32Ki
        nn.Conv2d(256, 128, (1, 1), (1, 1), bias=False),
        nn.ReLU(),
        # kernel memory = 64Ki
        nn.Conv2d(128, 128, (2, 2), (2, 2), bias=False),
        nn.ReLU(),
        # kernel memory = 32Ki
        nn.Conv2d(128, 256, (1, 1), (1, 1), bias=False),
        nn.ReLU(),
        # flatten
        sl.SumPool2d(kernel_size=(2, 2)),
        nn.Flatten(),
        # kernel memory = 16Ki
        nn.Linear(1024, 16, bias=False),
        nn.ReLU(),
    ]
)

SNN_KERNEL_MEM_TEST = DynapcnnNetwork(
    from_model(ann_for_kernel_mem_test, batch_size=1).spiking_model,
    discretize=True,
    input_shape=(1, 64, 64),
)

# Create a model which uses all neuron memories
ann_for_neuron_mem_test = nn.Sequential(
    *[
        # neuron memory = 32Ki
        nn.Conv2d(1, 8, (1, 1), (1, 1), bias=False),
        nn.ReLU(),
        # neuron memory = 64Ki
        nn.Conv2d(8, 16, (1, 1), (1, 1), bias=False),
        nn.ReLU(),
        # neuron memory = 32Ki
        nn.Conv2d(16, 128, (4, 4), (4, 4), bias=False),
        nn.ReLU(),
        # neuron memory = 16Ki
        nn.Conv2d(128, 64, (1, 1), (1, 1), bias=False),
        nn.ReLU(),
        # flatten
        sl.SumPool2d(kernel_size=(2, 2)),
        nn.Flatten(),
        # neron memory = 0.004Ki
        nn.Linear(4096, 4, bias=False),
        nn.ReLU(),
    ]
)

SNN_NEURON_MEM_TEST = DynapcnnNetwork(
    from_model(ann_for_neuron_mem_test, batch_size=1).spiking_model,
    discretize=True,
    input_shape=(1, 64, 64),
)


def test_auto_mapping():
    devices = ["speck2cmini", "speck2dmini"]

    for test_device in devices:
        # test weights/kernel memory mapping
        _ = SNN_KERNEL_MEM_TEST.make_config(layer2core_map="auto", device=test_device)
        assert SNN_KERNEL_MEM_TEST.layer2core_map == {0: 0, 1: 1, 2: 3, 3: 2, 4: 4}

        # test neuron memory mapping
        _ = SNN_NEURON_MEM_TEST.make_config(layer2core_map="auto", device=test_device)
        assert SNN_NEURON_MEM_TEST.layer2core_map == {0: 2, 1: 0, 2: 1, 3: 4, 4: 3}


def test_manual_mapping():
    devices = ["speck2cmini", "speck2dmini"]

    for test_device in devices:
        # test weights/kernel memory mapping
        layer2core_map = {0: 4, 1: 2, 2: 3, 3: 1, 4: 0}
        _ = SNN_KERNEL_MEM_TEST.make_config(
            layer2core_map=layer2core_map, device=test_device
        )
        assert SNN_KERNEL_MEM_TEST.layer2core_map == layer2core_map

        # test neuron memory mapping
        chip_layers_order = {0: 1, 1: 0, 2: 2, 3: 3, 4: 4}
        _ = SNN_NEURON_MEM_TEST.make_config(
            layer2core_map=chip_layers_order, device=test_device
        )
        assert SNN_NEURON_MEM_TEST.layer2core_map == chip_layers_order
