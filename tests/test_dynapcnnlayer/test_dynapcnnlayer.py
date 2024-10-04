# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

import pytest
from conftest_dynapcnnlayer import args_DynapcnnLayer

from sinabs.backend.dynapcnn.utils import (construct_dynapcnnlayer,
                                           construct_layerhandler,
                                           update_nodes_io)
from sinabs.backend.dynapcnn.weight_rescaling_methods import rescale_method_1


@pytest.mark.parametrize(
    "nodes_to_dcnnl_map, dpcnnl_idx, sinabs_edges, entry_point, expected_output",
    args_DynapcnnLayer,
)
def test_DynapcnnLayer(
    nodes_to_dcnnl_map, dpcnnl_idx, sinabs_edges, entry_point, expected_output
):
    """Tests the instantiation of a set of `DynapcnnLayer` belonging to the same SNN and the data computed
    within their constructors and shared among the differntly interacting instances (according to the graph
    described by `sinabs_edges`).
    """

    # create a `DynapcnnLayer` from the set of layers in `nodes_to_dcnnl_map[dpcnnl_idx]`.
    layerhandler = construct_layerhandler(
        dpcnnl_idx,
        True,
        sinabs_edges,
        nodes_to_dcnnl_map,
        rescale_method_1,
        entry_point,
    )
    dynapcnnlayer = construct_dynapcnnlayer(layerhandler)

    # check if any node (layer) in `dynapcnnlayer` has been modified (e.g. `nn.Linear` turned `nn.Conv2d`).
    node, output_shape = layerhandler.get_modified_node_io(
        nodes_to_dcnnl_map[dpcnnl_idx]
    )

    # one of the layers in `dynapcnnlayer` had its type modified (update input shape of nodes receiving from it).
    if isinstance(node, int) and isinstance(output_shape, tuple):
        update_nodes_io(node, output_shape, nodes_to_dcnnl_map, sinabs_edges)

    dpcnnl_index = expected_output[dpcnnl_idx]["dpcnnl_index"]
    conv_node_id = expected_output[dpcnnl_idx]["conv_node_id"]
    conv_in_shape = expected_output[dpcnnl_idx]["conv_in_shape"]
    conv_out_shape = expected_output[dpcnnl_idx]["conv_out_shape"]
    spk_node_id = expected_output[dpcnnl_idx]["spk_node_id"]
    pool_node_id = expected_output[dpcnnl_idx]["pool_node_id"]
    conv_rescaling_factor = expected_output[dpcnnl_idx]["conv_rescaling_factor"]
    dynapcnnlayer_destination = expected_output[dpcnnl_idx]["dynapcnnlayer_destination"]
    nodes_destinations = expected_output[dpcnnl_idx]["nodes_destinations"]
    entry_point = expected_output[dpcnnl_idx]["entry_point"]

    assert (
        layerhandler.dpcnnl_index == expected_output[dpcnnl_idx]["dpcnnl_index"]
    ), f"wrong 'DynapcnnLayer.dpcnnl_index': ID of the instance should be {dpcnnl_index}."
    assert (
        layerhandler.conv_node_id == expected_output[dpcnnl_idx]["conv_node_id"]
    ), f"wrong 'DynapcnnLayer.conv_node_id': convolution layer should be node {conv_node_id}."
    assert (
        layerhandler.conv_in_shape == expected_output[dpcnnl_idx]["conv_in_shape"]
    ), f"wrong 'DynapcnnLayer.conv_in_shape': input tensor shape of convolution should be {conv_in_shape}."
    assert (
        layerhandler.conv_out_shape == expected_output[dpcnnl_idx]["conv_out_shape"]
    ), f"wrong 'DynapcnnLayer.conv_out_shape': output tensor shape of convolution should be {conv_out_shape}."
    assert (
        layerhandler.spk_node_id == expected_output[dpcnnl_idx]["spk_node_id"]
    ), f"wrong 'DynapcnnLayer.spk_node_id': spiking layer should be node {spk_node_id}."
    assert (
        layerhandler.pool_node_id == expected_output[dpcnnl_idx]["pool_node_id"]
    ), f"wrong 'DynapcnnLayer.pool_node_id': pooling layer node(s) should be {pool_node_id}."
    assert (
        layerhandler.conv_rescaling_factor
        == expected_output[dpcnnl_idx]["conv_rescaling_factor"]
    ), f"wrong 'DynapcnnLayer.conv_rescaling_factor': computed re-scaling factor should be {conv_rescaling_factor}."
    assert (
        layerhandler.dynapcnnlayer_destination
        == expected_output[dpcnnl_idx]["dynapcnnlayer_destination"]
    ), f"wrong 'DynapcnnLayer.dynapcnnlayer_destination': the DynapcnnLayer(s) set as destination(s) should be {dynapcnnlayer_destination}."
    assert (
        layerhandler.nodes_destinations
        == expected_output[dpcnnl_idx]["nodes_destinations"]
    ), f"wrong 'DynapcnnLayer.nodes_destinations': the targeted nodes within other DynapcnnLayer instance(s) should be {nodes_destinations}."
    assert (
        layerhandler.entry_point == expected_output[dpcnnl_idx]["entry_point"]
    ), f"wrong 'DynapcnnLayer.entry_point': its value should be {entry_point}."
