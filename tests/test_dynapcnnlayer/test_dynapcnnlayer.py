import pytest
from sinabs.backend.dynapcnn import DynapcnnLayer
from conftest_dynapcnnlayer import args_DynapcnnLayer
import sys

sys.path.append('../../sinabs/backend/dynapcnn')

from weight_rescaling_methods import rescale_method_1
from utils import convert_Avg_to_Sum_pooling

@pytest.mark.parametrize("nodes_to_dcnnl_map, dpcnnl_idx, sinabs_edges, expected_output", args_DynapcnnLayer)
def test_DynapcnnLayer(nodes_to_dcnnl_map, dpcnnl_idx, sinabs_edges, expected_output):

    convert_Avg_to_Sum_pooling(nodes_to_dcnnl_map[dpcnnl_idx], sinabs_edges, nodes_to_dcnnl_map)

    dynapcnnlayer = DynapcnnLayer(
        dpcnnl_index        = dpcnnl_idx,
        dcnnl_data          = nodes_to_dcnnl_map[dpcnnl_idx],
        discretize          = True,
        sinabs_edges        = sinabs_edges,
        weight_rescaling_fn = rescale_method_1
    )

    config = {
        'dpcnnl_index': dynapcnnlayer.dpcnnl_index,
        'conv_node_id': dynapcnnlayer.conv_node_id,
        'conv_in_shape': dynapcnnlayer.conv_in_shape,
        'conv_out_shape': dynapcnnlayer.conv_out_shape,
        'spk_node_id': dynapcnnlayer.spk_node_id,
        'pool_node_id': dynapcnnlayer.pool_node_id,
        'conv_rescaling_factor': dynapcnnlayer.conv_rescaling_factor,
        'dynapcnnlayer_destination': dynapcnnlayer.dynapcnnlayer_destination,
        'nodes_destinations': dynapcnnlayer.nodes_destinations,
    }

    assert config == expected_output[dpcnnl_idx]