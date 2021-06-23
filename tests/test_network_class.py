from sinabs.from_torch import from_model
from torch import nn
import torch
import numpy as np


ann = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=(3, 3), bias=False),
    nn.ReLU(),
)

network = from_model(ann)
data = torch.rand((1, 1, 4, 4))


def test_reset_states():
    network(data)
    assert network.spiking_model[1].state.sum() != 0
    network.reset_states()
    assert network.spiking_model[1].state.sum() == 0


def test_compare_activations():
    analog, rates = network.compare_activations(data)
    assert len(analog) == len(rates) == 3
    for anl, spk in zip(analog, rates):
        assert np.squeeze(anl).shape == np.squeeze(spk).shape


def test_plot_comparison():
    """
    Test whether the plot_comparison() method of the sinabs.network.Network class
    could plot a nested-network which is not defined by torch.nn.Sequential(*module_list) directly.
    """
    class NestedANN(nn.Module):
        def __init__(self):
            super(NestedANN, self).__init__()
            seq = [nn.Conv2d(2, 8, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),
                   nn.ReLU(),
                   nn.AvgPool2d(kernel_size=(2, 2)),
                   nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                   nn.ReLU(),
                   nn.AvgPool2d(kernel_size=(2, 2)),
                   nn.Dropout2d(0.5),
                   nn.Conv2d(16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                   nn.ReLU(),
                   nn.AvgPool2d(kernel_size=(2, 2)),
                   nn.Flatten(),
                   nn.Dropout2d(0.5),
                   nn.Linear(8 * 8 * 8, 11, bias=True),
                   nn.ReLU(), ]
            self.main = nn.Sequential(*seq)

        def forward(self, x):
            return self.main(x)

    ann = NestedANN()
    input_shape = (2, 128, 128)
    sinabs_model = from_model(ann, input_shape=input_shape)
    example_input_ts = torch.rand([1, 2, 128, 128])
    # get the names of all spiking layers
    spiking_layers_names = [name for name, module in ann.named_modules() if isinstance(module, nn.ReLU)]
    num_spiking_layer = len(spiking_layers_names)

    ann_activation, snn_activation = sinabs_model.plot_comparison(example_input_ts)
    # if name_list arg not given, it will plot all spiking layer and 1 input layer
    assert len(ann_activation) == (num_spiking_layer + 1)
    assert len(snn_activation) == (num_spiking_layer + 1)
    # if pass a name_list arg, it will plot all given layer's comparison
    ann_activation, snn_activation = sinabs_model.plot_comparison(example_input_ts, spiking_layers_names)
    assert len(ann_activation) == len(spiking_layers_names)
    assert len(snn_activation) == len(spiking_layers_names)
