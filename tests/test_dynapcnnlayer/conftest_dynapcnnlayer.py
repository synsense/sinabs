from .model_dummy_1 import dcnnl_map_1, expected_output_1
from .model_dummy_2 import dcnnl_map_2, expected_output_2
from .model_dummy_3 import dcnnl_map_3, expected_output_3
from .model_dummy_4 import dcnnl_map_4, expected_output_4

# Args: dcnnl_map, discretize, expected_output
args_DynapcnnLayer = [
    (dcnnl_map_1, True, expected_output_1),
    (dcnnl_map_1, False, expected_output_1),
    (dcnnl_map_2, True, expected_output_2),
    (dcnnl_map_2, False, expected_output_2),
    (dcnnl_map_3, True, expected_output_3),
    (dcnnl_map_3, False, expected_output_3),
    (dcnnl_map_4, True, expected_output_4),
    (dcnnl_map_4, False, expected_output_4),
]
