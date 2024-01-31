import time

import pytest
import samna
import torch
from hw_utils import (
    find_open_devices,
    get_ones_network,
    is_any_samna_device_connected,
    is_device_connected,
)
from torch import nn

from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.io import calculate_neuron_address, neuron_address_to_cxy
from sinabs.layers import IAFSqueeze


def test_neuron_address_calculation():
    feature_map_shape = (16, 32, 32)  # channel, height, width

    for x in range(0, feature_map_shape[2]):
        for y in range(0, feature_map_shape[1]):
            for c in range(0, feature_map_shape[0]):
                address = calculate_neuron_address(x, y, c, feature_map_shape)

                recover_c, recover_x, recover_y = neuron_address_to_cxy(
                    address, feature_map_shape
                )
                assert c == recover_c
                assert x == recover_x
                assert y == recover_y

                pass


def test_neuron_leak_config():
    snn = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, 1), bias=True),
        IAFSqueeze(min_v_mem=-1.0, spike_threshold=1.0, batch_size=1),
        nn.Conv2d(
            in_channels=2, out_channels=8, kernel_size=(1, 1), stride=(2, 2), bias=True
        ),
        IAFSqueeze(min_v_mem=-1.0, spike_threshold=1.0, batch_size=1),
        nn.Conv2d(
            in_channels=8, out_channels=16, kernel_size=(1, 1), stride=(2, 2), bias=True
        ),
        IAFSqueeze(min_v_mem=-1.0, spike_threshold=1.0, batch_size=1),
    )

    # generate samna config object based on the torch.nn.Module
    dynapcnn = DynapcnnNetwork(
        snn=snn, discretize=True, dvs_input=True, input_shape=(1, 64, 64)
    )
    samna_cfg = dynapcnn.make_config(device="speck2fmodule")
    chip_layers_order = dynapcnn.chip_layers_ordering

    for lyr, channel_num in zip(chip_layers_order, [2, 8, 16]):
        assert samna_cfg.cnn_layers[lyr].leak_enable is True
        assert len(samna_cfg.cnn_layers[lyr].biases) == channel_num


@pytest.mark.skipif(
    not is_any_samna_device_connected(), reason="No samna device found!"
)
def test_neuron_leak():
    devices = find_open_devices()

    read_neuron_event_type = {
        "speck2fmodule": samna.speck2f.event.ReadNeuronValue,
        "speck2fdevkit": samna.speck2f.event.ReadNeuronValue,
        "speck2e": samna.speck2e.event.ReadNeuronValue,
        "speck2edevkit": samna.speck2e.event.ReadNeuronValue,
        "speck2b": samna.speck2b.event.ReadNeuronValue,
        "dynapcnndevkit": samna.dynapcnn.event.ReadNeuronValue,
    }

    snn = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=True),
        IAFSqueeze(min_v_mem=-1.0, spike_threshold=1.0, batch_size=1),
    )

    # set artificial values for vmem and bias
    weight_value = 1.0
    bias_value = -0.1
    vmem_value = 1.0
    snn[0].weight.data = torch.ones_like(snn[0].weight.data) * weight_value
    snn[0].bias.data = torch.ones_like(snn[0].bias.data) * bias_value
    # init the v_mem
    _ = snn(torch.zeros(1, 1, 4, 4))
    snn[1].v_mem.data = torch.ones_like(snn[1].v_mem.data) * vmem_value

    for device_name, device_info in devices.items():
        dynapcnn = DynapcnnNetwork(
            snn=snn, discretize=True, dvs_input=False, input_shape=(1, 4, 4)
        )
        dynapcnn.to(device=device_name, slow_clk_frequency=1)
        dynapcnn.samna_device.get_model().apply_configuration(dynapcnn.samna_config)
        time.sleep(1.0)
        input_events = []
        # create ReadNeuronValue event as input
        for x in range(4):
            for y in range(4):
                ev = read_neuron_event_type[device_name]()
                ev.layer = 0
                ev.address = calculate_neuron_address(
                    x=x, y=y, c=0, feature_map_size=(1, 4, 4)
                )
                input_events.append(ev)

        # check if neuron v_mem decrease along with time passes
        neuron_states = dict()
        for iter_times in range(5):
            # write input
            dynapcnn.samna_input_buffer.write(input_events)
            time.sleep(1.01)
            # get outputs
            output_events = dynapcnn.samna_output_buffer.get_events()

            for out_ev in output_events:
                c, x, y = neuron_address_to_cxy(
                    out_ev.address, feature_map_size=(1, 4, 4)
                )
                pre_neuron_state = neuron_states.get((c, x, y), 127)
                assert (
                    pre_neuron_state > out_ev.neuron_state
                ), "Neuron V_Mem doesn't decrease!"
                neuron_states.update({(c, x, y): out_ev.neuron_state})
                print(f"c:{c}, x:{x}, y:{y}, vmem:{out_ev.neuron_state}")
            print("--------")
