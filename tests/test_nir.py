import nir
import torch
import torch.nn as nn

import sinabs.layers as sl
from sinabs import from_nir, to_nir


def test_iaf():
    batch_size = 2
    iaf = sl.IAFSqueeze(batch_size=batch_size)
    graph = to_nir(iaf, torch.randn(batch_size, 10))
    converted = from_nir(graph, batch_size=batch_size)

    assert len(graph.nodes) == 1 + 2
    assert isinstance(graph.nodes["model"], nir.IF)
    assert len(graph.edges) == 0 + 2
    assert iaf.batch_size == converted.model.batch_size


def test_conv2d():
    batch_size = 2
    conv2d = nn.Conv2d(1, 3, 2)
    graph = to_nir(conv2d, torch.randn(batch_size, 1, 32, 32))
    converted = from_nir(graph, batch_size=batch_size)

    assert len(graph.nodes) == 1 + 2
    assert isinstance(graph.nodes["model"], nir.Conv2d)
    assert len(graph.edges) == 0 + 2
    assert conv2d.kernel_size == converted.model.kernel_size
    assert conv2d.stride == converted.model.stride
    assert conv2d.padding == converted.model.padding
    assert conv2d.dilation == converted.model.dilation


def test_from_sequential_to_nir():
    m = nn.Sequential(
        torch.nn.Linear(10, 2),
        sl.ExpLeak(tau_mem=10.0),
        sl.LIF(tau_mem=10.0),
        torch.nn.Linear(2, 1),
    )
    graph = to_nir(m, torch.randn(1, 10))
    assert len(graph.nodes) == 4 + 2
    assert isinstance(graph.nodes["0"], nir.Affine)
    assert isinstance(graph.nodes["1"], nir.LI)
    assert isinstance(graph.nodes["2"], nir.LIF)
    assert isinstance(graph.nodes["3"], nir.Affine)
    assert len(graph.edges) == 3 + 2


def test_from_linear_to_nir():
    in_features = 2
    out_features = 3
    m = torch.nn.Linear(in_features, out_features, bias=False)
    m2 = torch.nn.Linear(in_features, out_features, bias=True)
    graph = to_nir(m, torch.randn(1, in_features))
    assert len(graph.nodes) == 1 + 2
    assert graph.nodes["model"].weight.shape == (out_features, in_features)
    assert graph.nodes["model"].bias.shape == m2.bias.shape


def test_from_nir_to_sequential():
    batch_size = 4

    orig_model = nn.Sequential(
        torch.nn.Linear(10, 2),
        sl.ExpLeakSqueeze(tau_mem=10.0, batch_size=batch_size),
        sl.LIFSqueeze(tau_mem=10.0, batch_size=batch_size),
        torch.nn.Linear(2, 1),
    )
    nir_graph = to_nir(orig_model, torch.randn(batch_size, 10))

    converted_model = from_nir(nir_graph, batch_size=batch_size)
    converted_modules = list(converted_model.children())
    assert len(orig_model) + 2 == len(
        converted_modules
    )  # Addition of input and output modules
    torch.testing.assert_allclose(orig_model[0].weight, converted_modules[1].weight)
    torch.testing.assert_allclose(orig_model[0].bias, converted_modules[1].bias)
    assert type(orig_model[1]) == type(converted_modules[2])
    assert type(orig_model[2]) == type(converted_modules[3])
    torch.testing.assert_allclose(orig_model[3].weight, converted_modules[4].weight)
    torch.testing.assert_allclose(orig_model[3].bias, converted_modules[4].bias)


def test_as_pair():
    from sinabs.nir import _as_pair

    assert (2, 2) == _as_pair(2)
    assert (-1, 4) == _as_pair((-1, 4))


def test_2dcnn_network():
    from sinabs.nir import from_nir, to_nir

    orig_model = nn.Sequential(
        nn.Conv2d(2, 8, kernel_size=3, padding=1),
        sl.LIFSqueeze(tau_mem=10.0, batch_size=1),
        sl.SumPool2d(2),
        nn.Flatten(),
        nn.Linear(2 * 10 * 10, 5),
    )

    nir_graph = to_nir(orig_model, torch.rand(1, 2, 10, 10))

    loaded_model = from_nir(nir_graph, batch_size=1)
