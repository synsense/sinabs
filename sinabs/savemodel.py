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

from .network import Network
import pandas as pd
import json


def to_json(model: Network) -> str:
    """
    Returns a json descriptor of the model
    """
    summary_dataframe = model.summary()
    params_list = []
    for indx, row in summary_dataframe.iterrows():
        summary_dict = dict(row)
        params_dict = dict()
        # Layer number
        params_dict["Layer"] = indx
        params_dict["Class"] = summary_dict["Type"]
        # Padding
        if pd.isnull(summary_dict["Padding"]):
            pass
        else:
            params_dict["Padding_x"] = summary_dict["Padding"][0]
            params_dict["Padding_y"] = summary_dict["Padding"][2]

        # Pooling
        if pd.isnull(summary_dict["Pooling"]):
            pass
        else:
            params_dict["Pooling_x"] = summary_dict["Pooling"][1]
            params_dict["Pooling_y"] = summary_dict["Pooling"][0]

        # Stride
        if pd.isnull(summary_dict["Stride"]):
            pass
        else:
            params_dict["Stride_x"] = summary_dict["Stride"][1]
            params_dict["Stride_y"] = summary_dict["Stride"][0]

        # Kernel
        vKernelSize = summary_dict["Kernel"]
        if pd.isnull(vKernelSize):
            pass
        else:
            params_dict["Kernal_size_x"] = summary_dict["Kernel"][1]
            params_dict["Kernal_size_y"] = summary_dict["Kernel"][0]

        # Input feature number
        params_dict["Channels_in"] = summary_dict["Input_Shape"][0]

        # Output Features
        params_dict["Features_out"] = summary_dict["Output_Shape"][
            0
        ]  # Assuming channels first
        params_dict["Features_out_x"] = summary_dict["Output_Shape"][
            2
        ]  # Assuming channels first
        params_dict["Features_out_y"] = summary_dict["Output_Shape"][
            1
        ]  # Assuming channels first

        # Leak
        params_dict["Leak_enable"] = summary_dict["Bias_Params"] != 0

        # This index does not correspond to the indexing of layers on the chip
        # since there is no explicit layer for subsampling
        params_dict["Layer_out"] = {
            "nLayer": summary_dict["Layer_output"],
            "Feature_shift": 0,
        }
        params_list.append(params_dict)

    return json.dumps(params_list)
