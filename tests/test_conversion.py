import pytest
import torch.nn as nn

import sinabs
import sinabs.layers as sl


def test_layer_replacement_sequential():
    model = nn.Sequential(
        nn.Conv2d(2, 8, 3),
        sl.IAF(spike_threshold=2.0),
        nn.Conv2d(8, 16, 3),
        sl.IAF(spike_threshold=1.0),
        nn.Flatten(),
        nn.Linear(512, 10),
    )

    mapper_fn = lambda module: sl.IAFSqueeze(**module._param_dict, batch_size=18)
    filtered_model = sinabs.conversion.replace_module(
        model, sl.IAF, mapper_fn=mapper_fn
    )

    assert len(model) == len(filtered_model)
    assert type(filtered_model[1]) == sl.IAFSqueeze
    assert filtered_model[1].batch_size == 18
    assert filtered_model[1].spike_threshold == 2
    assert filtered_model[3].spike_threshold == 1
    assert model is not filtered_model


def test_layer_replacement_arbitrary():
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 8, 3)
            self.spike1 = sl.IAF(spike_threshold=2.0)
            self.conv2 = nn.Conv2d(8, 16, 3)
            self.spike2 = sl.IAF(spike_threshold=1.0)

    model = MyModel()
    mapper_fn = lambda module: sl.IAFSqueeze(**module._param_dict, batch_size=18)
    filtered_model = sinabs.conversion.replace_module(
        model, sl.IAF, mapper_fn=mapper_fn
    )

    assert type(filtered_model.spike1) == sl.IAFSqueeze
    assert filtered_model.spike1.batch_size == 18
    assert filtered_model.spike1.spike_threshold == 2
    assert filtered_model.spike2.spike_threshold == 1
    assert model is not filtered_model


def test_layer_replacement_individual():
    layer = sl.IAF(spike_threshold=2.0)
    mapper_fn = lambda module: sl.IAFSqueeze(**module.arg_dict, batch_size=4)
    squeezed_layer = sinabs.conversion.replace_module(layer, sl.IAF, mapper_fn)

    assert type(squeezed_layer) == sl.IAFSqueeze
    assert squeezed_layer.batch_size == 4
    assert squeezed_layer.spike_threshold == 2


def test_layer_replacement_individual_in_place():
    layer = sl.IAF(spike_threshold=2.0)
    mapper_fn = lambda module: sl.IAFSqueeze(**module.arg_dict, batch_size=4)
    with pytest.warns(UserWarning):
        squeezed_layer = sinabs.conversion.replace_module_(layer, sl.IAF, mapper_fn)

    assert type(layer) == sl.IAF
    assert squeezed_layer is None
