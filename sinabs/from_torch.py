import copy
from warnings import warn
import torch
from torch import nn
from sinabs.activation import MembraneSubtract

from sinabs import Network
import sinabs.layers as sl

_backends = {"sinabs": sl}

try:
    import sinabs.exodus.layers as el
except ModuleNotFoundError:
    pass
else:
    _backends["exodus"] = el


def from_model(
    model,
    input_shape=None,
    threshold=1.0,
    min_v_mem=-1.0,
    membrane_subtract=None,
    bias_rescaling=1.0,
    num_timesteps=None,
    batch_size=1,
    synops=False,
    add_spiking_output=False,
    backend="sinabs",
    kwargs_backend=None,
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
    :param min_v_mem: The lower bound of the potential in \
    convolutional and linear layers (same for all layers).
    :param membrane_subtract: Value subtracted from the potential upon \
    spiking for convolutional and linear layers (same for all layers).
    :param bias_rescaling: Biases are divided by this value.
    :param num_timesteps: Number of timesteps per sample. If None, `batch_size` \
    must be provided to seperate batch and time dimensions.
    :param batch_size: Must be provided if `num_timesteps` is None and is \
    ignored otherwise.
    :param synops: If True (default: False), register hooks for counting synaptic \
    operations during forward passes.
    :param add_spiking_output: If True (default: False), add a spiking layer \
    to the end of a sequential model if not present.
    :param backend: String defining the simulation backend (currently sinabs or exodus)
    :param kwargs_backend: Dict with additional kwargs for the simulation backend
    """
    return SpkConverter(
        input_shape=input_shape,
        threshold=threshold,
        min_v_mem=min_v_mem,
        membrane_subtract=membrane_subtract,
        bias_rescaling=bias_rescaling,
        batch_size=batch_size,
        num_timesteps=num_timesteps,
        synops=synops,
        add_spiking_output=add_spiking_output,
        backend=backend,
        kwargs_backend=None,
    ).convert(model)


class SpkConverter(object):
    """
    Converts a Torch model and returns a Sinabs network object.
    The modules in the model are analyzed, and a copy is returned, with all
    ReLUs, LeakyReLUs and NeuromorphicReLUs turned into SpikingLayers.

    :param input_shape: If provided, the layer dimensions are computed. \
    Otherwise they will computed at the first forward pass.
    :param threshold: The membrane potential threshold for spiking in \
    convolutional and linear layers (same for all layers).
    :param min_v_mem: The lower bound of the potential in \
    convolutional and linear layers (same for all layers).
    :param membrane_subtract: Value subtracted from the potential upon \
    spiking for convolutional and linear layers (same for all layers).
    :param bias_rescaling: Biases are divided by this value.
    :param num_timesteps: Number of timesteps per sample. If None, `batch_size` \
    must be provided to seperate batch and time dimensions.
    :param batch_size: Must be provided if `num_timesteps` is None and is \
    ignored otherwise.
    :param synops: If True (default: False), register hooks for counting synaptic \
    operations during foward passes.
    :param add_spiking_output: If True (default: False), add a spiking \
    layer to the end of a sequential model if not present.
    :param backend: String defining the simulation backend (currently sinabs or exodus)
    :param kwargs_backend: Dict with additional kwargs for the simulation backend
    """

    def __init__(
        self,
        input_shape=None,
        threshold=1.0,
        min_v_mem=-1.0,
        membrane_subtract=None,
        bias_rescaling=1.0,
        num_timesteps=None,
        batch_size=1,
        synops=False,
        add_spiking_output=False,
        backend="bptt",
        kwargs_backend=None,
    ):
        self.min_v_mem = min_v_mem
        self.threshold = threshold
        self.membrane_subtract = membrane_subtract
        self.bias_rescaling = bias_rescaling
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.synops = synops
        self.input_shape = input_shape
        self.add_spiking_output = add_spiking_output
        self.backend = backend
        self.kwargs_backend = kwargs_backend or dict()

    def relu2spiking(self):

        try:
            backend_module = _backends[self.backend]
        except KeyError:
            raise ValueError(
                f"Backend '{self.backend}' is not available. Available backends: "
                ", ".join(_backends.keys())
            )

        return backend_module.IAFSqueeze(
            spike_threshold=self.threshold,
            reset_fn=MembraneSubtract(),
            min_v_mem=self.min_v_mem,
            batch_size=self.batch_size,
            num_timesteps=self.num_timesteps,
            **self.kwargs_backend,
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
        network = Network(
            model,
            spk_model,
            input_shape=self.input_shape,
            synops=self.synops,
            batch_size=self.batch_size,
            num_timesteps=self.num_timesteps,
        )

        return network

    def convert_module(self, module):
        submodules = list(module.named_children())

        # iterate over the children
        for name, subm in submodules:
            # if it's one of the layers we're looking for, substitute it
            if isinstance(subm, (nn.ReLU, sl.NeuromorphicReLU)):
                setattr(module, name, self.relu2spiking())

            elif self.bias_rescaling != 1.0 and isinstance(
                subm, (nn.Linear, nn.Conv2d)
            ):
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
