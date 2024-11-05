# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

import pytest
import torch
from .conftest_dynapcnnnetwork import args_DynapcnnNetworkTest

from sinabs.backend.dynapcnn.dynapcnn_network import DynapcnnNetwork


@pytest.mark.parametrize(
    "snn, input_shape, batch_size, expected_output", args_DynapcnnNetworkTest
)
def test_DynapcnnNetwork(snn, input_shape, batch_size, expected_output):
    """Tests if the correct graph representing the connections between each DynapcnnLayer within a DynapcnnNetwork
    is created; if the DynapcnnLayer instances requiring input from a `Merge` are correctly flagged (along with what
    their arguments should be); if the correct topological order of the DynapcnnLayers (i.e., the order in which their
    forward methods should be called) is computed; if the output of the model matches what is expected.
    """

    dcnnnet = DynapcnnNetwork(snn, input_shape, batch_size)

    torch.manual_seed(0)
    x = torch.randn((batch_size, *input_shape))
    output = dcnnnet(x)

    module = dcnnnet.dynapcnn_module
    assert (
        expected_output["dcnnl_edges"] == module._dynapcnnlayer_edges
    ), "wrong list of edges describing DynapcnnLayer connectivity."

    # Convert source lists to sets to ignore order
    source_map = {
        node: set(sources) for node, sources in module._node_source_map.items()
    }
    assert expected_output["node_source_map"] == source_map, "wrong node source map"

    # Convert destination lists to sets to ignore order
    destination_map = {
        node: set(dests) for node, dests in module._destination_map.items()
    }
    assert (
        expected_output["destination_map"] == destination_map
    ), "wrong destination map"

    assert expected_output["entry_points"] == module._entry_points, "wrong entry points"

    assert expected_output["sorted_nodes"] == module._sorted_nodes, "wrong node sorting"

    assert (
        expected_output["output_shape"] == output.shape
    ), "wrong model output tensor shape."
