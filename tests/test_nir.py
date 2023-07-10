import nir
import torch
import torch.nn as nn

import sinabs.layers as sl
from sinabs import from_nir, to_nir


def test_from_sequential_to_nir():
    m = nn.Sequential(
        torch.nn.Linear(10, 2),
        sl.ExpLeak(tau_mem=10.0),
        sl.LIF(tau_mem=10.0),
        torch.nn.Linear(2, 1),
    )
    graph = to_nir(m, torch.randn(1, 10))
    assert len(graph.nodes) == 4
    assert isinstance(graph.nodes[0], nir.Linear)
    assert isinstance(graph.nodes[1], nir.LI)
    assert isinstance(graph.nodes[2], nir.LIF)
    assert isinstance(graph.nodes[3], nir.Linear)
    assert len(graph.edges) == 3


def test_from_linear_to_nir():
    in_features = 2
    out_features = 3
    m = torch.nn.Linear(in_features, out_features, bias=False)
    m2 = torch.nn.Linear(in_features, out_features, bias=True)
    graph = to_nir(m, torch.randn(1, in_features))
    assert len(graph.nodes) == 1
    assert graph.nodes[0].weights.shape == (out_features, in_features)
    assert graph.nodes[0].bias.shape == m2.bias.shape


def test_from_nir_to_sequential():
    batch_size = 4

    orig_model = nn.Sequential(
        torch.nn.Linear(10, 2),
        sl.ExpLeakSqueeze(tau_mem=10.0, batch_size=batch_size),
        sl.LIFSqueeze(tau_mem=10.0, batch_size=batch_size),
        torch.nn.Linear(2, 1),
    )
    nir_graph = to_nir(orig_model, torch.randn(batch_size, 10))

    convert_model = from_nir(nir_graph, batch_size=batch_size)

    assert len(orig_model) == len(convert_model)
    torch.testing.assert_allclose(orig_model[0].weight, convert_model[0].weight)
    torch.testing.assert_allclose(orig_model[0].bias, convert_model[0].bias)
    assert type(orig_model[1]) == type(convert_model[1])
    assert type(orig_model[2]) == type(convert_model[2])
    torch.testing.assert_allclose(orig_model[3].weight, convert_model[3].weight)
    torch.testing.assert_allclose(orig_model[3].bias, convert_model[3].bias)
