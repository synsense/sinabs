# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

import pytest
from conftest_dynapcnnlayer import args_DynapcnnLayer

from sinabs.backend.dynapcnn.utils import construct_dynapcnnlayers_from_mapper


@pytest.mark.parametrize(
    "dcnnl_map, discretize, expected_output",
    args_DynapcnnLayer,
)
def test_DynapcnnLayer(dcnnl_map, discretize, expected_output):
    """Tests the instantiation of a set of `DynapcnnLayer` belonging to the same SNN and the data computed
    within their constructors and shared among the differntly interacting instances (according to the graph
    described by `sinabs_edges`).
    """

    # create a `DynapcnnLayer` from the set of layers in `nodes_to_dcnnl_map[dpcnnl_idx]`.
    dynapcnn_layers, layer_handlers = construct_dynapcnnlayers_from_mapper(
        dcnnl_map=dcnnl_map, discretize=discretize, rescale_fn=None
    )

    for layer_index, dynapcnn_layer in dynapcnn_layers.items():

        # Test layer instance
        in_shape = expected_output[layer_index]["input_shape"]
        pool = expected_output[layer_index]["pool"]
        rescale_weights = expected_output[layer_index]["rescale_factor"]

        assert (
            tuple(dynapcnn_layer.in_shape) == in_shape
        ), f"wrong 'DynapcnnLayer.in_shape': Should be {in_shape}."
        assert (
            dynapcnn_layer.discretize == discretize
        ), f"wrong 'DynapcnnLayer.discretize': Should be {discretize}."
        in_shape = expected_output[layer_index]["input_shape"]
        assert (
            dynapcnn_layer.pool == pool
        ), f"wrong 'DynapcnnLayer.pool': Should be {pool}."
        in_shape = expected_output[layer_index]["input_shape"]
        assert (
            dynapcnn_layer.rescale_weights == rescale_weights
        ), f"wrong 'DynapcnnLayer.in_shape': Should be {rescale_weights}."

        # Test entries in layer info that are not directly repeated in layer or handler instances
        layer_info = dcnnl_map[layer_index]
        rescale_factors = expected_output[layer_index]["rescale_factors"]

        assert (
            layer_info["rescale_factors"] == rescale_factors
        ), f"wrong 'rescale_factors' entry: Should be {rescale_factors}."

        # Test layer handler instance
        layerhandler = layer_handlers[layer_index]
        destination_indices = expected_output[layer_index]["destination_indices"]
        entry_node = expected_output[layer_index]["entry_node"]

        assert (
            layerhandler.layer_index == layer_index
        ), f"wrong 'DynapcnnLayerHandler.layer_index': ID of the instance should be {layer_index}."
        assert (
            layerhandler.destination_indices
            == expected_output[layer_index]["destination_indices"]
        ), f"wrong 'DynapcnnLayerHandler.destination_indices': the DynapcnnLayer(s) set as destination(s) should be {destination_indices}."
        assert (
            layerhandler.entry_node == expected_output[layer_index]["entry_node"]
        ), f"wrong 'DynapcnnLayerHandler.entry_node': its value should be {entry_node}."
        assert layerhandler.assigned_core is None
