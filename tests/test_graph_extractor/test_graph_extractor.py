# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

import pytest
from sinabs.backend.dynapcnn.nir_graph_extractor import NIRtoDynapcnnNetworkGraph

from conftest_graph_extractor import args_GraphExtractor

@pytest.mark.parametrize("snn, input_dummy, expected_output", args_GraphExtractor)
def test_GraphExtractor(snn, input_dummy, expected_output):
    """ Tests the graph extraction from the original SNN being turned into a `DynapcnnNetwork`. These tests
    verify the correct functionality of the `NIRtoDynapcnnNetworkGraph` class, which implements the first pre-processing
    step on the conversion of the SNN into a DynapcnnNetwork.
    """

    graph_tracer = NIRtoDynapcnnNetworkGraph(snn, input_dummy)

    assert expected_output['edges'] == graph_tracer.edges, \
        f'wrong list of edges extracted from the SNN.'
    assert expected_output['name_2_indx_map'] == graph_tracer.name_2_indx_map, \
        f'wrong mapping from layer variable name to node ID.'
    assert expected_output['entry_nodes'] == graph_tracer.entry_nodes, \
        f'wrong list with entry node\'s IDs (i.e., layers serving as input to the SNN).'
    assert expected_output['nodes_io_shapes'] == graph_tracer.nodes_io_shapes, \
        f'wrong I/O shapes computed for one or more nodes.'
