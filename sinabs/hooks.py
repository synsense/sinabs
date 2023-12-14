from torch import nn
import torch


def _extract_single_input(input_):
    if len(input_) != 1:
        raise ValueError("Multiple inputs not supported for `input_diff_hook`")
    return = input_[0]

def conv_connection_map(layer, input_shape, output_shape):
    deconvole = nn.ConvTranspose2d(
        layer.out_channels,
        layer.in_channels,
        kernel_size=layer.kernel_size,
        padding=layer.padding,
        stride=layer.stride,
        dilation=layer.dilation,
        groups=layer.groups,
        bias=False,
    )
    deconvolve.weight.data.fill_(1)
    deconvole.weight.requires_grad = False
    output_ones = torch.ones(output_shape)
    connection_map = deconvolve(output_ones, output_size=input_shape).detach()
    connection_map.requires_grad = False
    return connection_map

def input_diff_hook(self, input_, output):
    input_ = _extract_single_input(input_)

    # Difference between absolute output and output with absolute weights
    if isinstance(self, nn.Conv2d):
        abs_weight_output = torch.nn.functional.conv2d(
            input=input_,
            weight=torch.abs(self.weight),
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )
    else:
        abs_weight_output = nn.functional.linear(
            input=input_,
            weight=torch.abs(self.weight),
        )
    self.diff_output = abs_weight_output - torch.abs(output)


def firing_rate_hook(self, input_, output):
    self.firing_rate = output.mean()

def firing_rate_per_neuron_hook(self, input_, output):
    self.firing_rate_per_neuron = output.mean((0,1))

def conv_layer_synops_hook(self, input_, output):
    input_ = _extract_single_input(input_)
    if (
        not hasattr(self, "connection_map")
        or self.connection_map.shape != input_.shape
    ):
        self.connection_map = conv_connection_map(self, input_.shape, output.shape)
    self.synops_raw = (input_ * self.connection_map).mean(0).sum()

def linear_layer_synops_hook(self, input_, output):
    input_ = _extract_single_input(input_)
    self.synops_raw = input_.mean(0).sum() * self.out_features


def model_synops_hook(self, input_, output):
    scale_factors = []
    for module in self:
        if isinstance(module, nn.AvgPool2d):
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
            scaling = ks**2 if isinstance(ks, int) else ks[0] * ks[1]
            scale_factors.append(scaling)
        if hasattr(module, "weight"):
            if hasattr(module, "synops_raw"):
                # Multiply all scale factors (or use 1 if empty)
                scaling = reduce(lambda x, y: x*y, scale_factors, 1)
                module.synops = module.synops_raw * scaling
            # For any module with weight: Reset `scale_factors` even if it doesn't count synops
            scale_factors = []

