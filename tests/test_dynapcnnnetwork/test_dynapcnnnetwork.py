import pytest, sys
import torch
from conftest_dynapcnnnetwork import args_NIRtoDynapcnnNetwork_edges_list, args_test_NIRtoDynapcnnNetwork_IO, args_DynapcnnLyers_edges_list, args_DynapcnnNetwork_forward_edges

sys.path.append('../../sinabs/backend/dynapcnn')

@pytest.mark.parametrize("snn, edges_list", args_NIRtoDynapcnnNetwork_edges_list)
def test_NIRtoDynapcnnNetwork_edges_list(snn, edges_list):
    from NIRGraphExtractor import NIRtoDynapcnnNetworkGraph

    batch_size = 1
    channels = 2
    height = 34
    width = 34

    input_shape = (batch_size, channels, height, width)
    dummy_input = torch.randn(input_shape)

    graph_tracer = NIRtoDynapcnnNetworkGraph(spiking_model = snn, dummy_input = dummy_input)

    assert graph_tracer.get_edges_list() == edges_list

@pytest.mark.parametrize("snn, io_dict", args_test_NIRtoDynapcnnNetwork_IO)
def test_NIRtoDynapcnnNetwork_IO(snn, io_dict):
    from NIRGraphExtractor import NIRtoDynapcnnNetworkGraph

    batch_size = 1
    channels = 2
    height = 34
    width = 34

    input_shape = (batch_size, channels, height, width)
    dummy_input = torch.randn(input_shape)

    graph_tracer = NIRtoDynapcnnNetworkGraph(spiking_model = snn, dummy_input = dummy_input)

    computed_IOs = {}

    for node, IO in io_dict.items():
        _in, _out = graph_tracer.get_node_io_shapes(node)

        computed_IOs[node] = {'in': _in, 'out': _out}

    assert computed_IOs == io_dict

@pytest.mark.parametrize("snn, dcnnl_edges_list", args_DynapcnnLyers_edges_list)
def test_DynapcnnLyers_edges_list(snn, dcnnl_edges_list):
    from sinabs.backend.dynapcnn import DynapcnnNetworkGraph

    channels = 2
    height = 34
    width = 34

    input_shape = (channels, height, width)

    hw_model = DynapcnnNetworkGraph(
        snn,
        discretize=True,
        input_shape=input_shape
    )

    computed_edges_list = hw_model.get_dynapcnnlayers_edges()

    assert computed_edges_list == dcnnl_edges_list

@pytest.mark.parametrize("snn, forward_edges_list", args_DynapcnnNetwork_forward_edges)
def test_DynapcnnNetwork_forward_edges(snn, forward_edges_list):
    from sinabs.backend.dynapcnn import DynapcnnNetworkGraph

    channels = 2
    height = 34
    width = 34

    input_shape = (channels, height, width)

    hw_model = DynapcnnNetworkGraph(
        snn,
        discretize=True,
        input_shape=input_shape
    )

    computed_forward_edges_list = hw_model.get_network_module().get_forward_edges()

    assert computed_forward_edges_list == forward_edges_list