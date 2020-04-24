import copy
from torch import nn
import sinabs.layers as sl
from sinabs import Network
import logging

logging.basicConfig(level=logging.DEBUG)


def from_model(model, input_shape, input_conversion_layer=False,
               threshold=1.0, threshold_low=-1.0, membrane_subtract=None,
               exclude_negative_spikes=False, bias_rescaling=1.0,
               all_2d_conv=False, batch_size=1):
    """
    Converts a Torch model and returns a Sinabs network object.
    Only sequential models or module lists are supported, with unpredictable
    behaviour on non-sequential models. This feature currently has limited
    capability. Supported layers are: Conv2d, AvgPool2d, MaxPool2d, Linear,
    BatchNorm2d (only if just after Linear or Conv2d), ReLU, Flatten,
    ZeroPad2d. LeakyReLUs are turned into ReLUs. Non-native torch layers
    supported are QuantizeLayer, YOLOLayer, NeuromorphicReLU, and
    DynapSumPoolLayer.

    :param model: a Torch model
    :param input_shape: the shape of the expected input
    :param input_conversion_layer: a Sinabs layer to be appended at the \
    beginning of the resulting network (typically Img2SpikeLayer or similar)
    :param threshold: The membrane potential threshold for spiking in \
    convolutional and linear layers (same for all layers).
    :param threshold_low: The lower bound of the potential in \
    convolutional and linear layers (same for all layers).
    :param membrane_subtract: Value subtracted from the potential upon \
    spiking for convolutional and linear layers (same for all layers).
    :param bias_rescaling: Biases are divided by this value.
    :param all_2d_conv: Whether to convert Flatten and Linear layers to convolutions.
    :return: :class:`.network.Network`
    """
    return SpkConverter(
        input_shape=input_shape,
        input_conversion_layer=input_conversion_layer,
        threshold=threshold,
        threshold_low=threshold_low,
        membrane_subtract=membrane_subtract,
        exclude_negative_spikes=exclude_negative_spikes,
        bias_rescaling=bias_rescaling,
        all_2d_conv=all_2d_conv,
        batch_size=batch_size,
    ).convert(model)


class SpkConverter(object):
    def __init__(self, input_shape, input_conversion_layer=False,
                 threshold=1.0, threshold_low=-1.0, membrane_subtract=None,
                 exclude_negative_spikes=False, bias_rescaling=1.0,
                 all_2d_conv=False, batch_size=1):
        """
        Converts a Torch model and returns a Sinabs network object.
        Only sequential models or module lists are supported, with unpredictable
        behaviour on non-sequential models. This feature currently has limited
        capability. Supported layers are: Conv2d, AvgPool2d, MaxPool2d, Linear,
        BatchNorm2d (only if just after Linear or Conv2d), ReLU, Flatten,
        ZeroPad2d. LeakyReLUs are turned into ReLUs. Non-native torch layers
        supported are QuantizeLayer, YOLOLayer, NeuromorphicReLU, and
        DynapSumPoolLayer.

        :param model: a Torch model
        :param input_shape: the shape of the expected input
        :param input_conversion_layer: a Sinabs layer to be appended at the \
        beginning of the resulting network (typically Img2SpikeLayer or similar)
        :param threshold: The membrane potential threshold for spiking in \
        convolutional and linear layers (same for all layers).
        :param threshold_low: The lower bound of the potential in \
        convolutional and linear layers (same for all layers).
        :param membrane_subtract: Value subtracted from the potential upon \
        spiking for convolutional and linear layers (same for all layers).
        :param bias_rescaling: Biases are divided by this value.
        :param all_2d_conv: Whether to convert Flatten and Linear layers to convolutions.
        """
        self.previous_layer_shape = input_shape
        self.threshold_low = threshold_low
        self.threshold = threshold
        self.membrane_subtract = membrane_subtract
        self.exclude_negative_spikes = exclude_negative_spikes
        self.bias_rescaling = bias_rescaling
        self.all_2d_conv = all_2d_conv
        self.batch_size = batch_size

        if input_conversion_layer:
            self.add("input_conversion", input_conversion_layer)

    def relu2spiking(self):
        return sl.SpikingLayerBPTT(
            input_shape= (0, 0, 0),  # TODO!
            threshold=self.threshold,
            threshold_low=self.threshold_low,
            membrane_subtract= self.threshold,  # TODO!
            layer_name="spiking",
            negative_spikes=not self.exclude_negative_spikes,
            batch_size=self.batch_size,
        )

    def convert(self, model):
        """
        Converts the Torch model and returns a Sinabs network object.

        :returns network: the Sinabs network object created by conversion.
        """
        spk_model = copy.deepcopy(model)

        logging.debug("## ORIGINAL MODEL")
        logging.debug(spk_model)
        self.convert_module(spk_model)
        logging.debug("## CONVERTED MODEL")
        logging.debug(spk_model)

        network = Network()
        network.spiking_model = spk_model
        network.analog_model = model

        return network

    def convert_module(self, module):
        if hasattr(module, "__len__"):
            # if sequential or similar, we iterate over it to access them by index
            submodules = enumerate(module)
        else:
            # otherwise, we look at the named_children and access by name
            submodules = list(module.named_children())

        # iterate over the children
        for name, subm in submodules:
            # if it's one of the layers we're looking for, substitute it
            if isinstance(subm, (nn.ReLU, sl.NeuromorphicReLU)):
                module[name] = self.relu2spiking()

            # if in turn it has children, go iteratively inside
            elif len(list(subm.named_children())):
                self.convert_module(subm)

            # otherwise we have a base layer of the non-interesting ones
            else:
                pass
