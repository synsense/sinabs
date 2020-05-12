import copy
from torch import nn
import sinabs.layers as sl
from sinabs import Network
import warnings
from numpy import product


def synops_hook(layer, inp, out):
    assert len(inp) == 1, "Multiple inputs not supported for synops hook"
    inp = inp[0]
    layer.tot_in = inp.sum().item()
    layer.tot_out = out.sum().item()
    layer.synops = layer.tot_in * layer.fanout
    layer.tw = inp.shape[0]


def from_model(model, input_shape=None, input_conversion_layer=False,
               threshold=1.0, threshold_low=-1.0, membrane_subtract=None,
               exclude_negative_spikes=False, bias_rescaling=1.0,
               all_2d_conv=False, batch_size=1, synops=True):
    """
    Converts a Torch model and returns a Sinabs network object.
    The modules in the model are analyzed, and a copy with the following
    substitutions is returned:
    - ReLUs, LeakyReLUs and NeuromorphicReLUs are turned into SpikingLayers
    - ...

    :param model: a Torch model
    :param input_shape: No effect. Backward compatibility only.
    :param input_conversion_layer: No longer supported.
    :param threshold: The membrane potential threshold for spiking in \
    convolutional and linear layers (same for all layers).
    :param threshold_low: The lower bound of the potential in \
    convolutional and linear layers (same for all layers).
    :param membrane_subtract: Value subtracted from the potential upon \
    spiking for convolutional and linear layers (same for all layers).
    :param bias_rescaling: Biases are divided by this value.
    :param all_2d_conv: Whether to convert Flatten and Linear layers to \
    convolutions. Currently not supported.
    :param synops: If True (default), register hooks for counting synaptic \
    operations during foward passes.
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
        synops=synops,
    ).convert(model)


class SpkConverter(object):
    def __init__(self, input_shape=None, input_conversion_layer=False,
                 threshold=1.0, threshold_low=-1.0, membrane_subtract=None,
                 exclude_negative_spikes=False, bias_rescaling=1.0,
                 all_2d_conv=False, batch_size=1, synops=True):
        """
        Converts a Torch model and returns a Sinabs network object.
        The modules in the model are analyzed, and substitutions are made:
        - ReLUs, LeakyReLUs and NeuromorphicReLUs are turned into SpikingLayers
        - ...

        :param input_shape: No effect. Backward compatibility only.
        :param input_conversion_layer: No longer supported.
        :param threshold: The membrane potential threshold for spiking in \
        convolutional and linear layers (same for all layers).
        :param threshold_low: The lower bound of the potential in \
        convolutional and linear layers (same for all layers).
        :param membrane_subtract: Value subtracted from the potential upon \
        spiking for convolutional and linear layers (same for all layers).
        :param bias_rescaling: Biases are divided by this value.
        :param all_2d_conv: Whether to convert Flatten and Linear layers to \
        convolutions. Currently not supported.
        :param synops: If True (default), register hooks for counting synaptic \
        operations during foward passes.
        """
        if input_shape is not None:
            warnings.warn("Input shape is now determined automatically and has no effect")
        if all_2d_conv:  # TODO
            raise NotImplementedError("Turning linear into conv not supported yet.")
        if input_conversion_layer is not False:
            raise NotImplementedError("Input conversion layer no longer supported.")

        self.threshold_low = threshold_low
        self.threshold = threshold
        self.membrane_subtract = membrane_subtract
        self.exclude_negative_spikes = exclude_negative_spikes
        self.bias_rescaling = bias_rescaling
        # self.all_2d_conv = all_2d_conv
        self.batch_size = batch_size
        self.synops = synops

        if input_conversion_layer:
            self.add("input_conversion", input_conversion_layer)

    def relu2spiking(self):
        return sl.SpikingLayerBPTT(
            threshold=self.threshold,
            threshold_low=self.threshold_low,
            membrane_subtract=self.membrane_subtract,
            layer_name="spiking",
            negative_spikes=False,
            batch_size=self.batch_size,
        )

    def convert(self, model):
        """
        Converts the Torch model and returns a Sinabs network object.

        :returns network: the Sinabs network object created by conversion.
        """
        spk_model = copy.deepcopy(model)

        # import logging
        # logging.debug("## ORIGINAL MODEL")
        # logging.debug(spk_model)
        self.convert_module(spk_model)
        # logging.debug("## CONVERTED MODEL")
        # logging.debug(spk_model)

        network = Network()
        device = next(model.parameters()).device
        network.spiking_model = spk_model.to(device)
        network.analog_model = model

        return network

    def convert_module(self, module):
        if hasattr(module, "__len__"):
            # if sequential or similar, we iterate over it to access by index
            submodules = enumerate(module)
        else:
            # otherwise, we look at the named_children and access by name
            submodules = list(module.named_children())

        # iterate over the children
        for name, subm in submodules:
            # if it's one of the layers we're looking for, substitute it
            if isinstance(subm, (nn.ReLU, sl.NeuromorphicReLU)):
                module[name] = self.relu2spiking()

            elif isinstance(subm, nn.Linear) and self.synops:
                subm.fanout = subm.out_features
                subm.register_forward_hook(synops_hook)
                if subm.bias is not None:
                    subm.bias.data = subm.bias.data.clone().detach() / self.bias_rescaling
            elif isinstance(subm, nn.Conv2d) and self.synops:
                subm.fanout = (
                    subm.out_channels
                    * product(subm.kernel_size)
                    / product(subm.stride)
                )
                subm.register_forward_hook(synops_hook)
                if subm.bias is not None:
                    subm.bias.data = subm.bias.data.clone().detach() / self.bias_rescaling

            # if in turn it has children, go iteratively inside
            elif len(list(subm.named_children())):
                self.convert_module(subm)

            # otherwise we have a base layer of the non-interesting ones
            else:
                pass  # yes this is useless but it's for clarity
