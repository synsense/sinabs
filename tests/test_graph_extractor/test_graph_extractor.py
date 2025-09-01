import pytest
from conftest_graph_extractor import args_GraphExtractor

from sinabs.backend.dynapcnn.nir_graph_extractor import GraphExtractor


def fix_node_ids(expected_output, graph_extractor):
    """Match node IDs between graph extractor and expected output

    Node IDs can be assigned in many ways. This function prevents test
    errors from generated IDs not matching expected output

    Parameters
    ----------
    expected_output: Dict with expected output
    graph_extractor: GraphExtractor instance

    Returns
    -------
    Expected outputs with remapped node IDs
    """
    idx_map = {
        expected_idx: graph_extractor.name_2_indx_map[name]
        for name, expected_idx in expected_output["name_2_indx_map"].items()
    }
    edges = {(idx_map[src], idx_map[tgt]) for src, tgt in expected_output["edges"]}
    name_2_indx_map = {
        name: idx_map[idx] for name, idx in expected_output["name_2_indx_map"].items()
    }
    entry_nodes = {idx_map[node] for node in expected_output["entry_nodes"]}
    nodes_io_shapes = {
        idx_map[node]: shape
        for node, shape in expected_output["nodes_io_shapes"].items()
    }
    return {
        "edges": edges,
        "name_2_indx_map": name_2_indx_map,
        "entry_nodes": entry_nodes,
        "nodes_io_shapes": nodes_io_shapes,
    }

@pytest.mark.skip("Need NONSEQ update")
@pytest.mark.parametrize("snn, input_dummy, expected_output", args_GraphExtractor)
def test_GraphExtractor(snn, input_dummy, expected_output):
    """Tests the graph extraction from the original SNN being turned into a `DynapcnnNetwork`. These tests
    verify the correct functionality of the `GraphExtractor` class, which implements the first pre-processing
    step on the conversion of the SNN into a DynapcnnNetwork.
    """

    graph_tracer = GraphExtractor(snn, input_dummy)
    expected_output = fix_node_ids(expected_output, graph_tracer)

    assert (
        expected_output["edges"] == graph_tracer.edges
    ), "wrong list of edges extracted from the SNN."
    assert (
        expected_output["name_2_indx_map"] == graph_tracer.name_2_indx_map
    ), "wrong mapping from layer variable name to node ID."
    assert (
        expected_output["entry_nodes"] == graph_tracer.entry_nodes
    ), "wrong list with entry node's IDs (i.e., layers serving as input to the SNN)."
    assert (
        expected_output["nodes_io_shapes"] == graph_tracer.nodes_io_shapes
    ), "wrong I/O shapes computed for one or more nodes."
