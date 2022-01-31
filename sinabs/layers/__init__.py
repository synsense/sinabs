from .crop2d import Cropping2dLayer
from .maxpool2d import SpikingMaxPooling2dLayer
from .input_layer import InputLayer
from .quantize import QuantizeLayer
from .neuromorphic_relu import NeuromorphicReLU
from .sumpool2d import SumPool2d
from .img_to_spk import Img2SpikeLayer
from .sig_to_spk import Sig2SpikeLayer
from .stateful_layer import StatefulLayer
from .iaf import IAF, IAFRecurrent, IAFSqueeze
from .lif import LIF, LIFRecurrent, LIFSqueeze
from .alif import ALIF, ALIFRecurrent
from .exp_leak import ExpLeak, ExpLeakSqueeze
from .squeeze_layer import SqueezeMixin

try:
    from sinabs.slayer import layers as slayer_layers
except ModuleNotFoundError:
    pass
else:
    _layers_with_backend = (IAF, IAFSqueeze, LIF, LIFSqueeze, ExpLeak, ExpLeakSqueeze)

    for lyr in _layers_with_backend:
        # Find equivalent slayer layer classes by name
        lyr_slayer = getattr(slayer_layers, lyr.__name__)
        # Add sinabs layer class to slayer version's external backends
        if hasattr(lyr_slayer, "external_backends"):
            lyr_slayer.external_backends[lyr.backend] = lyr
        else:
            lyr_slayer.external_backends = {lyr.backend: lyr}
        # Add slayer version to sinabs layer class' external backends
        if hasattr(lyr, "external_backends"):
            lyr.external_backends[lyr_slayer.backend] = lyr_slayer
        else:
            lyr.external_backends = {lyr_slayer.backend: lyr_slayer}
