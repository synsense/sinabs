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

from .crop2d import Cropping2dLayer
from .maxpool2d import SpikingMaxPooling2dLayer
from .inputlayer import InputLayer
from .quantize import QuantizeLayer
from .neuromorphic_relu import NeuromorphicReLU
from .sumpool2d import SumPool2d
from .img_to_spk import Img2SpikeLayer
from .sig_to_spk import Sig2SpikeLayer
from .iaf_bptt import SpikingLayer
from .iaf_bptt import SpikingLayer as SpikingLayerBPTT

# Deprecated
from .deprecated.flatten import FlattenLayer
from .deprecated.iaf_conv1d import SpikingConv1dLayer
from .deprecated.iaf_conv2d import SpikingConv2dLayer
from .deprecated.iaf_convtranspose2d import SpikingConvTranspose2dLayer
from .deprecated.sumpool2d import SumPooling2dLayer
from .deprecated.iaf_conv3d import SpikingConv3dLayer
from .deprecated.zeropad2d import ZeroPad2dLayer
from .deprecated.iaf_linear import SpikingLinearLayer
from .deprecated.iaf_tc import SpikingTemporalConv1dLayer
from .deprecated.iaf import SpikingLayer as LegacySpikingLayer
from .deprecated.layer import Layer
from .deprecated.yolo import YOLOLayer
