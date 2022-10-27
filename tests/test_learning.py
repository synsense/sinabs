from sinabs.layers import NeuromorphicReLU
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn.dynapcnn_network import DynapcnnNetwork
import torch.nn as nn
import torch


class DynapCnnNetA(nn.Module):
    def __init__(self, quantize=False, n_out=1):
        super().__init__()

        seq = [
            # core 0
            nn.Conv2d(2, 16, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),
            nn.ReLU(),
            # core 1
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 2
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 7
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 4
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 5
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 6
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 256, kernel_size=(2, 2), padding=(0, 0), bias=False),
            nn.ReLU(),
            # core 3
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 128, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(),
            # core 8
            nn.Conv2d(128, n_out, kernel_size=(1, 1), padding=(0, 0), bias=False),
            nn.ReLU(),
            nn.Flatten(),  # otherwise torch complains
        ]

        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)

def test_learning():
    sdc = DynapCnnNetA()
    snn = from_model(sdc.seq, batch_size=1)
    print(snn)
    input_shape = (2, 128, 128)
    input_data = torch.rand((10, *input_shape)) * 1000
    dynapcnn_net = DynapcnnNetwork(snn, input_shape=input_shape, discretize=False, dvs_input=False)
    print(dynapcnn_net)

    optim = torch.optim.Adam(dynapcnn_net.parameters())
    criterion = torch.nn.MSELoss()
    for i in range(10):
        optim.zero_grad()
        dynapcnn_net.zero_grad()

        out = dynapcnn_net(input_data)
        loss = criterion(out.mean(), torch.tensor(1.0))
        loss.backward()

        optim.step()