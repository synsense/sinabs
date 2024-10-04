# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

from model_dummy_1 import expected_output as expected_output_1
from model_dummy_1 import input_dummy as input_dummy_1
from model_dummy_1 import snn as snn_1
from model_dummy_2 import expected_output as expected_output_2
from model_dummy_2 import input_dummy as input_dummy_2
from model_dummy_2 import snn as snn_2
from model_dummy_3 import expected_output as expected_output_3
from model_dummy_3 import input_dummy as input_dummy_3
from model_dummy_3 import snn as snn_3
from model_dummy_4 import expected_output as expected_output_4
from model_dummy_4 import input_dummy as input_dummy_4
from model_dummy_4 import snn as snn_4

args_GraphExtractor = [
    (snn_1, input_dummy_1, expected_output_1),
    (snn_2, input_dummy_2, expected_output_2),
    (snn_3, input_dummy_3, expected_output_3),
    (snn_4, input_dummy_4, expected_output_4),
]
