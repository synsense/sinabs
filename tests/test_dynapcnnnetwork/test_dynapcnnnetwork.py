# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

import pytest
import torch
from conftest_dynapcnnnetwork import args_DynapcnnNetworkTest

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

    assert (
        expected_output["dcnnl_edges"] == dcnnnet.dcnnl_edges
    ), f"wrong list of edges describing DynapcnnLayer connectivity."

    for node, args in dcnnnet.merge_points.items():

        assert (
            node in expected_output["merge_points"]
        ), f"DynapcnnLayer {node} is not a merge point."
        assert (
            args["sources"] == expected_output["merge_points"][node]["sources"]
        ), f"DynapcnnLayer {node} has wrong input sources ({args})."

    for entry_point in expected_output["entry_point"]:
        assert dcnnnet.layers_mapper[
            entry_point
        ].entry_point, f"DynapcnnLayer {entry_point} should be an entry point."

    assert (
        expected_output["topological_order"] == dcnnnet.topological_order
    ), f"wrong topological ordering between DynapcnnLayers."
    assert (
        expected_output["output_shape"] == output.shape
    ), f"wrong model output tensor shape."
