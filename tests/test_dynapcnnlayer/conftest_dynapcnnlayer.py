from model_dummy_1 import nodes_to_dcnnl_map_1, sinabs_edges_1, expected_output_1
from model_dummy_2 import nodes_to_dcnnl_map_2, sinabs_edges_2, expected_output_2
from model_dummy_3 import nodes_to_dcnnl_map_3, sinabs_edges_3, expected_output_3
from model_dummy_4 import nodes_to_dcnnl_map_4, sinabs_edges_4, expected_output_4

args_DynapcnnLayer = [
    (nodes_to_dcnnl_map_1, 0, sinabs_edges_1, [0], expected_output_1), 
    (nodes_to_dcnnl_map_1, 1, sinabs_edges_1, [0], expected_output_1), 
    (nodes_to_dcnnl_map_1, 2, sinabs_edges_1, [0], expected_output_1), 
    (nodes_to_dcnnl_map_1, 3, sinabs_edges_1, [0], expected_output_1), 
    (nodes_to_dcnnl_map_1, 4, sinabs_edges_1, [0], expected_output_1), 
    (nodes_to_dcnnl_map_2, 0, sinabs_edges_2, [0], expected_output_2), 
    (nodes_to_dcnnl_map_2, 1, sinabs_edges_2, [0], expected_output_2), 
    (nodes_to_dcnnl_map_2, 2, sinabs_edges_2, [0], expected_output_2), 
    (nodes_to_dcnnl_map_2, 3, sinabs_edges_2, [0], expected_output_2), 
    (nodes_to_dcnnl_map_2, 4, sinabs_edges_2, [0], expected_output_2), 
    (nodes_to_dcnnl_map_2, 5, sinabs_edges_2, [0], expected_output_2), 
    (nodes_to_dcnnl_map_2, 6, sinabs_edges_2, [0], expected_output_2), 
    (nodes_to_dcnnl_map_3, 0, sinabs_edges_3, [0, 8], expected_output_3), 
    (nodes_to_dcnnl_map_3, 1, sinabs_edges_3, [0, 8], expected_output_3), 
    (nodes_to_dcnnl_map_3, 2, sinabs_edges_3, [0, 8], expected_output_3), 
    (nodes_to_dcnnl_map_3, 3, sinabs_edges_3, [0, 8], expected_output_3), 
    (nodes_to_dcnnl_map_3, 4, sinabs_edges_3, [0, 8], expected_output_3), 
    (nodes_to_dcnnl_map_3, 5, sinabs_edges_3, [0, 8], expected_output_3), 
    (nodes_to_dcnnl_map_3, 6, sinabs_edges_3, [0, 8], expected_output_3), 
    (nodes_to_dcnnl_map_3, 7, sinabs_edges_3, [0, 8], expected_output_3), 
    (nodes_to_dcnnl_map_3, 8, sinabs_edges_3, [0, 8], expected_output_3), 
    (nodes_to_dcnnl_map_4, 0, sinabs_edges_4, [0], expected_output_4), 
    (nodes_to_dcnnl_map_4, 1, sinabs_edges_4, [0], expected_output_4), 
    (nodes_to_dcnnl_map_4, 2, sinabs_edges_4, [0], expected_output_4), 
    (nodes_to_dcnnl_map_4, 3, sinabs_edges_4, [0], expected_output_4), 
    (nodes_to_dcnnl_map_4, 4, sinabs_edges_4, [0], expected_output_4), 
    (nodes_to_dcnnl_map_4, 5, sinabs_edges_4, [0], expected_output_4), 
]