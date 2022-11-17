import torch
import torch.nn as nn

import sinabs
import sinabs.layers as sl


class SNN(nn.Module):
    def __init__(self, hidden_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            sl.IAF(),
            nn.Linear(hidden_dim, hidden_dim),
            sl.IAF(),
            nn.Linear(hidden_dim, hidden_dim),
            sl.IAF(),
            nn.Linear(hidden_dim, 5),
        )
        self.spike_output = sl.IAF()

    def forward(self, x):
        return self.spike_output(self.net(x))


def test_reset_states():
    model = SNN()
    assert model.net[1].v_mem.sum() == 0

    input = torch.rand((5, 10, 2))
    model(input)
    assert model.net[1].v_mem.grad_fn is not None
    assert model.net[1].v_mem.sum() != 0

    sinabs.reset_states(model)
    assert model.net[1].v_mem.grad_fn is None
    assert model.net[1].v_mem.sum() == 0


def test_zero_grad():
    model = SNN()
    assert model.net[1].v_mem.grad == None

    input = torch.rand((5, 10, 2))
    output = model(input)

    assert model.net[1].v_mem.grad_fn is not None
    assert model.net[1].v_mem.sum() != 0

    sinabs.zero_grad(model)
    assert model.net[1].v_mem.grad_fn is None
    assert model.net[1].v_mem.sum() != 0
