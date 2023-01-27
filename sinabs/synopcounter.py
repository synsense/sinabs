import warnings
from functools import partial

import torch
import torch.nn as nn
from numpy import product

import sinabs.layers as sl
from sinabs.layers import NeuromorphicReLU


def spiking_hook(self, input_, output):
    """Forward hook meant for Sinabs/Exodus layers.

    Calculates n_neurons (scalar), firing_rate_per_neuron (C,H,W) and average firing_rate (scalar).
    """
    if isinstance(self, sl.SqueezeMixin):
        output = output.reshape(self.batch_size, self.num_timesteps, *output.shape[1:])
    self.n_neurons = output[0, 0].numel()
    if self.firing_rate_per_neuron == None:
        self.firing_rate_per_neuron = output.mean((0, 1))
    else:
        self.firing_rate_per_neuron = self.firing_rate_per_neuron + output.mean((0, 1))
    self.tracked_firing_rate = self.tracked_firing_rate + output.mean()
    self.n_batches = self.n_batches + 1


def synops_hook(deconvolve, self, input_, output):
    """Forward hook for parameter layers such as Conv2d or Linear.

    Calculates the total amount of input and output and the number of synaptic operations. The
    `unflattened_shape` parameter is a bit of a hack since Sinabs sometimes uses squeezed inputs
    and from inside the hook we cannot know whether the input is squeezed or not.
    """
    assert len(input_) == 1, "Multiple inputs not supported for synops hook"
    input_ = input_[0]
    if self.unflattened_shape is not None:
        batch_size, num_timesteps = self.unflattened_shape
        input_ = input_.reshape(batch_size, num_timesteps, *input_.shape[1:])
        output = output.reshape(batch_size, num_timesteps, *output.shape[1:])
        self.num_timesteps = input_.shape[1]
        # for the purposes of counting synops, we can just sum over time and work a fixed shape from now on
        input_ = input_.sum(1)
        output = output.sum(1)
    else:
        self.num_timesteps = 0
    if isinstance(self, nn.Conv2d):
        with torch.no_grad():
            connection_map = deconvolve(torch.ones_like(output))
        self.synops = self.synops + (input_ * connection_map).sum()
    else:
        self.synops = self.synops + input_.sum() * self.fanout
    self.n_samples = self.n_samples + input_.shape[0]


class SNNAnalyzer:
    """Helper class to acquire statistics for spiking and parameter layers at the same time. To
    calculate the number of synapses between neurons accurately, a simple scaling factor based on
    the kernel size is not enough, as neurons on the edges of the input will have a different
    amount of connections as neurons in the center. This is why we make use of a transposed
    convolution layer to calculate this synaptic connection map once. The amount of synapses
    between two layers depends on all parameters of a conv layer such as kernel size, stride,
    groups etc. Transposed conv will take all those parameters into account and 'reproject' the
    output of a conv layer. As long as the spatial dimensions don't change during training, we can
    reuse the same connection map, which is a tensor of the same dimensions as the layer output. We
    can therefore calculate the number of synaptic operations accurately for each layer by
    multiplying the respective connection map with the output.

    Parameters:
        model: Your PyTorch model.
        dt: the number of milliseconds corresponding to a time step in the simulation (default 1.0).

    Example:
        >>> analyzer = SNNAnalyzer(my_spiking_model)
        >>> output = my_spiking_model(input_)  # forward pass
        >>> layer_stats = analyzer.get_layer_statistics()
        >>> model_stats = analyzer.get_model_statistics()
    """

    def __init__(self, model: torch.nn.Module, dt: float = 1.0):
        self.model = model
        self.dt = dt
        self.handles = []
        self._setup_hooks()

    def _setup_hooks(self):
        """Attaches spiking and parameter hooks to matching layers and resets all counters."""
        # This is a hack to unflatten inputs/outputs in conv/linear layers
        # inside the hook since we cannot know from the shape alone whether
        # it is flattened (B*T,C,H,W) or unflattened (B,T,C,H,W)
        unflattened_shape = None
        for layer in self.model.modules():
            if isinstance(layer, sl.SqueezeMixin):
                unflattened_shape = (layer.batch_size, layer.num_timesteps)

        for layer in self.model.modules():
            if isinstance(layer, sl.StatefulLayer):
                layer.firing_rate_per_neuron = None
                layer.tracked_firing_rate = 0
                layer.n_batches = 0
                handle = layer.register_forward_hook(spiking_hook)
                self.handles.append(handle)
            if isinstance(layer, torch.nn.Conv2d):
                deconvolve = nn.ConvTranspose2d(
                    layer.out_channels,
                    layer.in_channels,
                    kernel_size=layer.kernel_size,
                    padding=layer.padding,
                    stride=layer.stride,
                    dilation=layer.dilation,
                    groups=layer.groups,
                    bias=False,
                )
                deconvolve.weight.requires_grad = False
                deconvolve.weight.data = torch.ones_like(deconvolve.weight)
                layer.synops = 0
                layer.n_samples = 0
                layer.unflattened_shape = unflattened_shape
                handle = layer.register_forward_hook(partial(synops_hook, deconvolve))
                self.handles.append(handle)
            elif isinstance(layer, torch.nn.Linear):
                layer.fanout = layer.out_features
                layer.synops = 0
                layer.n_samples = 0
                layer.unflattened_shape = unflattened_shape
                handle = layer.register_forward_hook(partial(synops_hook, None))
                self.handles.append(handle)

    def __del__(self):
        for handle in self.handles:
            handle.remove()

    def get_layer_statistics(self) -> dict:
        """Outputs a dictionary with statistics for each individual layer.

        The statistics such as firing rate per neuron, the number of neurons or synops are averaged
        across batches.
        """
        spike_dict = {}
        spike_dict["spiking"] = {}
        spike_dict["parameter"] = {}
        scale_facts = []
        for name, module in self.model.named_modules():
            if (
                hasattr(module, "tracked_firing_rate")
                or hasattr(module, "firing_rate_per_neuron")
                or hasattr(module, "n_neurons")
            ):
                spike_dict["spiking"][name] = {}
            if hasattr(module, "tracked_firing_rate"):
                spike_dict["spiking"][name].update(
                    {"firing_rate": module.tracked_firing_rate / module.n_batches}
                )
            if hasattr(module, "firing_rate_per_neuron"):
                spike_dict["spiking"][name].update(
                    {
                        "firing_rate_per_neuron": module.firing_rate_per_neuron
                        / module.n_batches
                    }
                )
            if hasattr(module, "n_neurons"):
                spike_dict["spiking"][name].update({"n_neurons": module.n_neurons})

            if isinstance(module, torch.nn.AvgPool2d):
                # Average pooling scales down the number of counted synops due to the averaging.
                # We need to correct for that by accumulating the scaling factors and multiplying
                # them to the counted Synops in the next conv or linear layer
                if module.kernel_size != module.stride:
                    warnings.warn(
                        f"In order for the Synops counter to work accurately the pooling "
                        f"layers kernel size should match their strides. At the moment at layer {name}, "
                        f"the kernel_size = {module.kernel_size}, the stride = {module.stride}."
                    )
                ks = module.kernel_size
                scale_factor = ks**2 if isinstance(ks, int) else ks[0] * ks[1]
                scale_facts.append(scale_factor)
            if hasattr(module, "synops"):
                scale_factor = 1
                while len(scale_facts) != 0:
                    scale_factor *= scale_facts.pop()
                spike_dict["parameter"][name] = {
                    # "fanout_prev": module.fanout,
                    "synops": module.synops / module.n_samples * scale_factor,
                    "num_timesteps": module.num_timesteps,
                    "time_window": module.num_timesteps * self.dt,
                    "SynOps/s": (module.synops / module.n_samples * scale_factor)
                    / module.num_timesteps
                    / self.dt
                    * 1000,
                }
        return spike_dict

    def get_model_statistics(self):
        """Outputs a dictionary with statistics that are summarised across all layers.

        The statistics such as firing rate per neuron or synops are averaged across batches.
        """
        stats_dict = {}
        firing_rates = []
        synops = 0.0
        for name, module in self.model.named_modules():
            if hasattr(module, "firing_rate_per_neuron"):
                if module.n_batches > 0:
                    firing_rates.append(
                        module.firing_rate_per_neuron.ravel() / module.n_batches
                    )
                else:
                    firing_rates.append(torch.tensor([0.0]))
            if hasattr(module, "synops"):
                if module.n_samples > 0:
                    synops = synops + module.synops / module.n_samples
                else:
                    synops = 0
        if len(firing_rates) > 0:
            stats_dict["firing_rate"] = torch.cat(firing_rates).mean()
        stats_dict["synops"] = synops
        return stats_dict

    def reset(self):
        for handle in self.handles:
            handle.remove()
        self._setup_hooks()


class SynOpCounter:
    """Counter for the synaptic operations emitted by all Neuromorphic ReLUs in an analog CNN
    model.

    Parameters:
        modules: list of modules, e.g. MyTorchModel.modules()
        sum_activations: If True (default), returns a single number of synops, otherwise a list of layer synops.

    Example:
        >>> counter = SynOpCounter(MyTorchModel.modules(), sum_activations=True)
        >>> output = MyTorchModule(input)  # forward pass
        >>> synop_count = counter()
    """

    def __init__(self, modules, sum_activations=True):
        self.modules = []
        for module in modules:
            if isinstance(module, NeuromorphicReLU) and module.fanout > 0:
                self.modules.append(module)

        if len(self.modules) == 0:
            raise ValueError("No NeuromorphicReLU found in module list.")

        self.sum_activations = sum_activations

    def __call__(self):
        synops = []
        for module in self.modules:
            synops.append(module.activity)

        if self.sum_activations:
            synops = torch.stack(synops).sum()
        return synops
