import copy
from typing import Callable, Optional, Tuple, Union
from warnings import warn
import torch
from torch import nn
from sinabs.activation import MembraneSubtract, MultiSpike, SingleExponential

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
    spike_threshold=1.0,
    spike_fn: Callable = MultiSpike,
    reset_fn: Callable = MembraneSubtract(),
    surrogate_grad_fn: Callable = SingleExponential(),
    min_v_mem=-1.0,
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

    Parameters
    ----------
    model:
        Torch model
    input_shape:
        If provided, the layer dimensions are computed. \
        Otherwise they will be computed at the first forward pass.
    spike_threshold:
        The membrane potential threshold for spiking (same for all layers).
    spike_fn: Callable
        The spike dynamics to determine the number of spikes out
    reset_fn: Callable
        The reset mechanism of the neuron (like reset to zero, or subtract)
    surrogate_grad_fn: Callable
        The surrogate gradient method for the spiking dynamics
    min_v_mem:
        The lower bound of the potential in (same for all layers).
    bias_rescaling:
        Biases are divided by this value.
    num_timesteps:
        Number of timesteps per sample. If None, `batch_size` \
        must be provided to seperate batch and time dimensions.
    batch_size:
        Must be provided if `num_timesteps` is None and is \
        ignored otherwise.
    synops:
        If True (default: False), register hooks for counting synaptic \
        operations during forward passes.
    add_spiking_output:
        If True (default: False), add a spiking layer \
        to the end of a sequential model if not present.
    backend:
        String defining the simulation backend (currently sinabs or exodus)
    kwargs_backend:
        Dict with additional kwargs for the simulation backend
    """
    return SpkConverter(
        input_shape=input_shape,
        spike_threshold=spike_threshold,
        spike_fn=spike_fn,
        reset_fn=reset_fn,
        surrogate_grad_fn=surrogate_grad_fn,
        min_v_mem=min_v_mem,
        bias_rescaling=bias_rescaling,
        batch_size=batch_size,
        num_timesteps=num_timesteps,
        synops=synops,
        add_spiking_output=add_spiking_output,
        backend=backend,
        kwargs_backend=kwargs_backend,
    ).convert(model)


class SpkConverter(object):
    """
    Converts a Torch model and returns a Sinabs network object.
    The modules in the model are analyzed, and a copy is returned, with all
    ReLUs, LeakyReLUs and NeuromorphicReLUs turned into SpikingLayers.

    Parameters
    ----------

    input_shape:
        If provided, the layer dimensions are computed. \
        Otherwise they will computed at the first forward pass.
    spike_threshold:
        The membrane potential threshold for spiking layers (same for all layers).
    spike_fn: Callable
        The spike dynamics to determine the number of spikes out
    reset_fn: Callable
        The reset mechanism of the neuron (like reset to zero, or subtract)
    surrogate_grad_fn: Callable
        The surrogate gradient method for the spiking dynamics
    min_v_mem:
        The lower bound of the potential in \
        convolutional and linear layers (same for all layers).
    bias_rescaling:
        Biases are divided by this value.
    num_timesteps:
        Number of timesteps per sample. If None, `batch_size` \
        must be provided to seperate batch and time dimensions.
    batch_size:
        Must be provided if `num_timesteps` is None and is \
        ignored otherwise.
    synops:
        If True (default: False), register hooks for counting synaptic \
        operations during foward passes.
    add_spiking_output:
        If True (default: False), add a spiking \
        layer to the end of a sequential model if not present.
    backend:
        String defining the simulation backend (currently sinabs or exodus)
    kwargs_backend:
        Dict with additional kwargs for the simulation backend
    """

    def __init__(
        self,
        input_shape: Optional[Tuple] = None,
        spike_threshold=1.0,
        spike_fn: Callable = MultiSpike(),
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        min_v_mem: float = -1.0,
        bias_rescaling: float = 1.0,
        num_timesteps: Optional[int] = None,
        batch_size: int = 1,
        synops: bool = False,
        add_spiking_output: bool = False,
        backend: str = "bptt",
        kwargs_backend: dict = None,
    ):
        self.min_v_mem = min_v_mem
        self.spike_threshold = spike_threshold
        self.spike_fn = spike_fn
        self.reset_fn = reset_fn
        self.surrogate_grad_fn = surrogate_grad_fn
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
            spike_threshold=self.spike_threshold,
            spike_fn=self.spike_fn,
            reset_fn=self.reset_fn,
            surrogate_grad_fn=self.surrogate_grad_fn,
            min_v_mem=self.min_v_mem,
            batch_size=self.batch_size,
            num_timesteps=self.num_timesteps,
            **self.kwargs_backend,
        ).to(self.device)

    def convert(self, model: nn.Module) -> Network:
        """
        Converts the Torch model and returns a Sinabs network object.
        Parameters
        ----------
        model:
            A torch module.

        Returns
        -------
        network:
            The Sinabs network object created by conversion.
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
