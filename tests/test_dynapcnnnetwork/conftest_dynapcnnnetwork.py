# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

from model_dummy_1 import batch_size as batch_size_1
from model_dummy_1 import expected_output as expected_output_1
from model_dummy_1 import input_shape as input_shape_1
from model_dummy_1 import snn as snn_1
from model_dummy_2 import batch_size as batch_size_2
from model_dummy_2 import expected_output as expected_output_2
from model_dummy_2 import input_shape as input_shape_2
from model_dummy_2 import snn as snn_2
from model_dummy_3 import batch_size as batch_size_3
from model_dummy_3 import expected_output as expected_output_3
from model_dummy_3 import input_shape as input_shape_3
from model_dummy_3 import snn as snn_3
from model_dummy_4 import batch_size as batch_size_4
from model_dummy_4 import expected_output as expected_output_4
from model_dummy_4 import input_shape as input_shape_4
from model_dummy_4 import snn as snn_4

args_DynapcnnNetworkTest = [
    (snn_1, input_shape_1, batch_size_1, expected_output_1),
    (snn_2, input_shape_2, batch_size_2, expected_output_2),
    (snn_3, input_shape_3, batch_size_3, expected_output_3),
    (snn_4, input_shape_4, batch_size_4, expected_output_4),
]
