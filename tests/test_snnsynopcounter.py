import torch
from torch import nn
from sinabs.synopcounter import SNNSynOpCounter
from sinabs.layers import SpikingLayer


def test_tinynetwork():
    model = nn.Sequential(
        nn.Conv2d(1, 5, kernel_size=2),
        SpikingLayer(),
    )

    inp = torch.tensor([[
        [[0, 0, 0],
         [0, 3, 0],
         [0, 0, 0]]
    ]]).float()

    counter = SNNSynOpCounter(model)
    model(inp)
    # 3 spikes, 2x2 kernel, 5 channels
    assert counter.get_synops()["SynOps"].sum() == 60
    assert counter.get_total_synops() == 60
