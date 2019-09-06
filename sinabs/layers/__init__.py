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

from .layer import TorchLayer
from .crop2d import Cropping2dLayer, from_cropping2d_keras_conf
from .dropout import from_dropout_keras_conf
from .flatten import FlattenLayer, from_flatten_keras_conf
from .iaf import SpikingLayer
from .iaf_conv1d import SpikingConv1dLayer
from .iaf_conv2d import (
    SpikingConv2dLayer,
    from_conv2d_keras_conf,
    from_dense_keras_conf,
)
from .iaf_conv3d import SpikingConv3dLayer

from .iaf_convtranspose2d import SpikingConvTranspose2dLayer

from .maxpool2d import SpikingMaxPooling2dLayer, from_maxpool2d_keras_conf

from .inputlayer import (
    InputLayer,
    get_input_shape_from_keras_conf,
    from_input_keras_conf,
)
from .quantize import QuantizeLayer
from .sumpool2d import (
    SumPooling2dLayer,
    from_sumpool2d_keras_conf,
    from_avgpool2d_keras_conf,
)
from .zeropad2d import ZeroPad2dLayer, from_zeropad2d_keras_conf
from .img_to_spk import Img2SpikeLayer

from .iaf_linear import SpikingLinearLayer
