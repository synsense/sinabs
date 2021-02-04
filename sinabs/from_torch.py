import copy
from warnings import warn
import torch
from torch import nn
import sinabs.layers as sl
from sinabs import Network
from numpy import product


def synops_hook(layer, inp, out):
    assert len(inp) == 1, "Multiple inputs not supported for synops hook"
    inp = inp[0]
    layer.tot_in = inp.sum().item()
    layer.tot_out = out.sum().item()
    layer.synops = layer.tot_in * layer.fanout
    layer.tw = inp.shape[0]


def from_model(
    model,
    input_shape=None,
    threshold=1.0,
    threshold_low=-1.0,
    membrane_subtract=None,
    bias_rescaling=1.0,
    batch_size=None,
    synops=True,
    add_spiking_output=False,
):
    """
    Converts a Torch model and returns a Sinabs network object.
    The modules in the model are analyzed, and a copy is returned, with all
    ReLUs, LeakyReLUs and NeuromorphicReLUs turned into SpikingLayers.

    :param model: a Torch model
    :param input_shape: If provided, the layer dimensions are computed. \
    Otherwise they will be computed at the first forward pass.
    :param threshold: The membrane potential threshold for spiking in \
    convolutional and linear layers (same for all layers).
    :param threshold_low: The lower bound of the potential in \
    convolutional and linear layers (same for all layers).
    :param membrane_subtract: Value subtracted from the potential upon \
    spiking for convolutional and linear layers (same for all layers).
    :param bias_rescaling: Biases are divided by this value.
    :param synops: If True (default), register hooks for counting synaptic \
    operations during foward passes.
    :param add_spiking_output: If True (default: False), add a spiking layer \
    to the end of a sequential model if not present.
    """
    return SpkConverter(
        input_shape=input_shape,
        threshold=threshold,
        threshold_low=threshold_low,
        membrane_subtract=membrane_subtract,
        bias_rescaling=bias_rescaling,
        batch_size=batch_size,
        synops=synops,
        add_spiking_output=add_spiking_output,
    ).convert(model)


class SpkConverter(object):
    def __init__(
        self,
        input_shape=None,
        threshold=1.0,
        threshold_low=-1.0,
        membrane_subtract=None,
        bias_rescaling=1.0,
        batch_size=None,
        synops=True,
        add_spiking_output=False,
    ):
        """
        Converts a Torch model and returns a Sinabs network object.
        The modules in the model are analyzed, and a copy is returned, with all
        ReLUs, LeakyReLUs and NeuromorphicReLUs turned into SpikingLayers.

        :param input_shape: If provided, the layer dimensions are computed. \
        Otherwise they will computed at the first forward pass.
        :param threshold: The membrane potential threshold for spiking in \
        convolutional and linear layers (same for all layers).
        :param threshold_low: The lower bound of the potential in \
        convolutional and linear layers (same for all layers).
        :param membrane_subtract: Value subtracted from the potential upon \
        spiking for convolutional and linear layers (same for all layers).
        :param bias_rescaling: Biases are divided by this value.
        :param synops: If True (default), register hooks for counting synaptic \
        operations during foward passes.
        :param add_spiking_output: If True (default: False), add a spiking \
        layer to the end of a sequential model if not present.
        """
        self.threshold_low = threshold_low
        self.threshold = threshold
        self.membrane_subtract = membrane_subtract
        self.bias_rescaling = bias_rescaling
        self.batch_size = batch_size
        self.synops = synops
        self.input_shape = input_shape
        self.add_spiking_output = add_spiking_output

    def relu2spiking(self):
        return sl.SpikingLayerBPTT(
            threshold=self.threshold,
            threshold_low=self.threshold_low,
            membrane_subtract=self.membrane_subtract,
            batch_size=self.batch_size,
        ).to(self.device)

    def convert(self, model):
        """
        Converts the Torch model and returns a Sinabs network object.

        :returns network: the Sinabs network object created by conversion.
        """
        spk_model = copy.deepcopy(model)
        # device is taken as the device of the first element of the input state_dict
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        if self.add_spiking_output:
            # Add spiking output to sequential model
            if isinstance(spk_model, nn.Sequential) and not isinstance(
                spk_model[-1], (nn.ReLU, sl.NeuromorphicReLU)
            ):
                spk_model.add_module("Spiking output", nn.ReLU())
            else:
                warn(
                    "Spiking output can only be added to sequential models that do not end in a ReLU. No layer has been added."
                )

        self.convert_module(spk_model)
        network = Network(model, spk_model, input_shape=self.input_shape)

        return network

    def convert_module(self, module):
        # if hasattr(module, "__len__"):
        #     # if sequential or similar, we iterate over it to access by index
        #     submodules = enumerate(module)
        # else:
        #     # otherwise, we look at the named_children and access by name
        submodules = list(module.named_children())

        # iterate over the children
        for name, subm in submodules:
            # if it's one of the layers we're looking for, substitute it
            if isinstance(subm, (nn.ReLU, sl.NeuromorphicReLU)):
                # module[name] = self.relu2spiking()
                setattr(module, name, self.relu2spiking())

            elif isinstance(subm, nn.Linear) and self.synops:
                subm.fanout = subm.out_features
                subm.register_forward_hook(synops_hook)
                if subm.bias is not None:
                    subm.bias.data = (
                        subm.bias.data.clone().detach() / self.bias_rescaling
                    )
            elif isinstance(subm, nn.Conv2d) and self.synops:
                subm.fanout = (
                    subm.out_channels * product(subm.kernel_size) / product(subm.stride)
                )
                subm.register_forward_hook(synops_hook)
                if subm.bias is not None:
                    subm.bias.data = (
                        subm.bias.data.clone().detach() / self.bias_rescaling
                    )

            # if in turn it has children, go iteratively inside
            elif len(list(subm.named_children())):
                self.convert_module(subm)

            # otherwise we have a base layer of the non-interesting ones
            else:
                pass  # yes this is useless but it's for clarity
