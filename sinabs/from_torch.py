from typing import Callable, Optional, Tuple, Type
from warnings import warn

import torch
from torch import nn

import sinabs
import sinabs.layers as sl
from sinabs.activation import MembraneSubtract, MultiSpike, SingleExponential
from sinabs.conversion import replace_module


def from_model(
    model: nn.Module,
    input_shape: Optional[Tuple[int, int, int]] = None,
    spike_threshold: torch.Tensor = torch.tensor(1.0),
    spike_fn: Callable = MultiSpike,
    reset_fn: Callable = MembraneSubtract(),
    surrogate_grad_fn: Callable = SingleExponential(),
    min_v_mem: float = -1.0,
    bias_rescaling: float = 1.0,
    batch_size: Optional[int] = None,
    num_timesteps: Optional[int] = None,
    synops: bool = False,
    add_spiking_output: bool = False,
    spike_layer_class: Type = sl.IAFSqueeze,
    backend=None,
    kwargs_backend: Optional[dict] = None,
):
    """Converts a Torch model and returns a Sinabs network object. The modules in the model are
    analyzed, and a copy is returned, with all ReLUs and NeuromorphicReLUs turned into
    SpikingLayers.

    Parameters:
        model: Torch model
        input_shape: If provided, the layer dimensions are computed. Otherwise they will be computed at the first forward pass.
        spike_threshold: The membrane potential threshold for spiking (same for all layers).
        spike_fn: The spike dynamics to determine the number of spikes out
        reset_fn: The reset mechanism of the neuron (like reset to zero, or subtract)
        surrogate_grad_fn: The surrogate gradient method for the spiking dynamics
        min_v_mem: The lower bound of the potential in (same for all layers).
        bias_rescaling: Biases are divided by this value.
        batch_size: Must be provided if `num_timesteps` is None and is ignored otherwise.
        num_timesteps: Number of timesteps per sample. If None, `batch_size` must be provided to seperate batch and time dimensions.
        synops: If True (default: False), register hooks for counting synaptic operations during forward passes.
        add_spiking_output: If True (default: False), add a spiking layer to the end of a sequential model if not present.
        spike_layer_class: Can be for example sinabs.layers.IAFSqueeze (default) or EXODUS equivalent.
        backend: String defining the simulation backend (currently sinabs or exodus)
        kwargs_backend: Dict with additional kwargs for the simulation backend
    """
    if backend is not None:
        warn(
            "The 'backend' argument is deprecated and will be removed in a future release, please use spike_layer_class instead."
        )
        _backends = {"sinabs": sl}
        try:
            import sinabs.exodus.layers as el
        except ModuleNotFoundError:
            pass
        else:
            _backends["exodus"] = el
        spike_layer_class = _backends[backend].IAFSqueeze

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    if kwargs_backend is None:
        kwargs_backend = dict()
    if issubclass(spike_layer_class, sl.SqueezeMixin):
        kwargs_backend["batch_size"] = batch_size
        kwargs_backend["num_timesteps"] = num_timesteps

    def mapper_fn(module):
        return spike_layer_class(
            spike_threshold=spike_threshold,
            spike_fn=spike_fn,
            reset_fn=reset_fn,
            surrogate_grad_fn=surrogate_grad_fn,
            min_v_mem=min_v_mem,
            **kwargs_backend,
        ).to(device)

    snn = replace_module(model=model, source_class=nn.ReLU, mapper_fn=mapper_fn)
    snn = replace_module(
        model=snn, source_class=sl.NeuromorphicReLU, mapper_fn=mapper_fn
    )

    if add_spiking_output:
        if isinstance(model, nn.Sequential) and not isinstance(
            model[-1], (nn.ReLU, sl.NeuromorphicReLU)
        ):
            snn.add_module(
                "spike_output",
                spike_layer_class(
                    spike_threshold=spike_threshold,
                    spike_fn=spike_fn,
                    reset_fn=reset_fn,
                    surrogate_grad_fn=surrogate_grad_fn,
                    min_v_mem=min_v_mem,
                    **kwargs_backend,
                ).to(device),
            )
        else:
            warn(
                "Spiking output can only be added to sequential models that do not end in a ReLU. No layer has been added."
            )

    for module in snn.modules():
        if bias_rescaling != 1.0 and isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, "bias") and module.bias is not None:
                with torch.no_grad():
                    module.bias.data /= bias_rescaling

    network = sinabs.network.Network(
        model,
        snn,
        input_shape=input_shape,
        synops=synops,
        batch_size=batch_size,
        num_timesteps=num_timesteps,
    )
    return network
