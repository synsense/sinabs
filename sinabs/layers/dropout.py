#  Copyright (c) 2019-2019     aiCTX AG (Sadique Sheik, Qian Liu).
#
#  This file is part of sinabs
#
#  sinabs is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sinabs is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with sinabs.  If not, see <https://www.gnu.org/licenses/>.

from typing import Tuple, Dict, List
import torch.nn as nn


def from_dropout_keras_conf(
    layer_config: Dict, input_shape: Tuple, spiking=False
) -> List:
    """
    Load Dropout  keras config file
    :param layer_config: configuration dictionary
    :param input_shape: input data shape to determine output dimensions
    :param spiking: bool True if spiking layer needs to be loaded
    """
    # Config depth consistency
    if "config" in layer_config:
        pass
    else:
        layer_config = {"config": layer_config}

    try:
        layer_name = layer_config["name"]
    except KeyError:
        layer_name = layer_config["config"]["name"]
    layer_list = []

    # Dropout rate
    dropout_rate = layer_config["config"]["rate"]

    if spiking:
        pass
    else:
        # Pooling layer initialization
        torch_analogue_layer = nn.Dropout(p=dropout_rate)
        layer_list.append((layer_name, torch_analogue_layer))

    return layer_list
