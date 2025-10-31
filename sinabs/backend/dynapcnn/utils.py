from collections import defaultdict, deque
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import warnings

import sinabs.layers as sl

from .crop2d import Crop2d
from .dvs_layer import DVSLayer
from .exceptions import InputConfigurationError

if TYPE_CHECKING:
    from sinabs.backend.dynapcnn.dynapcnn_network import DynapcnnNetwork

# Other than `COMPLETELY_IGNORED_LAYER_TYPES`, `IGNORED_LAYER_TYPES` are
# part of the graph initially and are needed to ensure proper handling of
# graph structure (e.g. Merge nodes) or meta-information (e.g.
# `nn.Flatten` for io-shapes)
COMPLETELY_IGNORED_LAYER_TYPES = (nn.Identity, nn.Dropout, nn.Dropout2d)
IGNORED_LAYER_TYPES = (nn.Flatten, sl.Merge)

Edge = Tuple[int, int]  # Define edge-type alias


def parse_device_id(device_id: str) -> Tuple[str, int]:
    """Parse device id into device type and device index.

    Args:
        device_id (str): Device id typically of the form `device_type:index`.
            In case no index is specified, the default index of zero is returned.

    Returns:
        Tuple[str, int]: (device_type, index) Returns a tuple with the index and device type.
    """
    parts = device_id.split(sep=":")
    if len(parts) == 1:
        device_type = parts[0]
        index = 0
    elif len(parts) == 2:
        device_type, index = parts
    else:
        raise Exception(
            "Device id not understood. A string of form `device_type:index` expected."
        )

    return device_type, int(index)


def get_device_id(device_type: str, index: int) -> str:
    """Generate a device id string given a device type and its index.

    Args:
        device_type (str): Device type
        index (int): Device index

    Returns:
        str: A string of the form `device_type:index`
    """
    return f"{device_type}:{index}"


def standardize_device_id(device_id: str) -> str:
    """Standardize device id string.

    Args:
        device_id (str): Device id string. Could be of the form `device_type` or `device_type:index`

    Returns:
        str: Returns a sanitized device id of the form `device_type:index`
    """
    device_type, index = parse_device_id(device_id=device_id)
    return get_device_id(device_type=device_type, index=index)


def topological_sorting(edges: Set[Tuple[int, int]]) -> List[int]:
    """Performs a topological sorting (using Kahn's algorithm) of a graph
    described by a list of edges. An entry node `X` of the graph have to be
    flagged inside `edges` by a tuple `('input', X)`.

    Args:
        edges (set): the edges describing the *acyclic* graph.

    Returns:
        The nodes sorted by the graph's topology.
    """

    graph = defaultdict(list)
    in_degree = defaultdict(int)

    # initialize the graph and in-degrees.
    for u, v in edges:
        if u != "input":
            graph[u].append(v)
            in_degree[v] += 1
        else:
            if v not in in_degree:
                in_degree[v] = 0
        if v not in in_degree:
            in_degree[v] = 0

    # find all nodes with zero in-degrees.
    zero_in_degree_nodes = deque(
        [node for node, degree in in_degree.items() if degree == 0]
    )

    # process nodes and create the topological order.
    topological_order = []

    while zero_in_degree_nodes:
        node = zero_in_degree_nodes.popleft()
        topological_order.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree_nodes.append(neighbor)

    # check if all nodes are processed (to handle cycles).
    if len(topological_order) == len(in_degree):
        return topological_order

    raise ValueError("The graph has a cycle and cannot be topologically sorted.")


def convert_cropping2dlayer_to_crop2d(
    layer: sl.Cropping2dLayer, input_shape: Tuple[int, int]
) -> Crop2d:
    """Convert a sinabs layer of type Cropping2dLayer to Crop2d layer.

    Args:
        layer: Cropping2dLayer.
        input_shape: (height, width) input dimensions.

    Returns:
        Equivalent Crop2d layer.
    """
    h, w = input_shape
    top = layer.top_crop
    left = layer.left_crop
    bottom = h - layer.bottom_crop
    right = w - layer.right_crop
    print(h, w, left, right, top, bottom, layer.right_crop, layer.bottom_crop)
    return Crop2d(((top, bottom), (left, right)))


WeightLayer = TypeVar("WeightLayer", nn.Linear, nn.Conv2d)


def merge_bn(
    weight_layer: WeightLayer, bn: Union[nn.BatchNorm1d, nn.BatchNorm2d]
) -> WeightLayer:
    """Merge a convolutional or linear layer with subsequent batch
    normalization.

    Args:
        weight_layer: torch.nn.Conv2d or nn.Linear. Convolutional or linear
            layer
        bn: torch.nn.Batchnorm2d or nn.Batchnorm1d. Batch normalization.

    Returns:
        Weight layer including batch normalization.
    """
    mu = bn.running_mean
    sigmasq = bn.running_var

    if bn.affine:
        gamma, beta = bn.weight, bn.bias
    else:
        gamma, beta = 1.0, 0.0

    factor = gamma / sigmasq.sqrt()

    weight = weight_layer.weight.data.clone().detach()
    bias = 0.0 if weight_layer.bias is None else weight_layer.bias.data.clone().detach()

    weight_layer = deepcopy(weight_layer)

    new_bias = beta + (bias - mu) * factor
    if weight_layer.bias is None:
        weight_layer.bias = nn.Parameter(new_bias)
    else:
        weight_layer.bias.data = new_bias

    for __ in range(weight_layer.weight.ndim - factor.ndim):
        factor.unsqueeze_(-1)
    weight_layer.weight.data = weight * factor

    return weight_layer


def merge_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Merge a convolutional layer with subsequent batch normalization.

    Args:
        conv: torch.nn.Conv2d. Convolutional layer.
        bn: torch.nn.Batchnorm2d. Batch normalization.

    Returns:
        Convolutional layer including batch normalization.
    """
    warnings.warn(
        "`merge_conv_bn` is deprecated. Use `merge_bn` instead.", DeprecationWarning
    )
    return merge_bn(conv, bn)


def extend_readout_layer(model: "DynapcnnNetwork") -> "DynapcnnNetwork":
    """Return a copied and extended model with the readout layer extended to 4 times the number of
    output channels. For Speck 2E and 2F, to get readout with correct output index, we need to
    extend the final layer to 4 times the number of output.

    Args:
        model (DynapcnnNetwork): the model to be extended

    Returns:
        DynapcnnNetwork: the extended model
    """
    model = deepcopy(model)
    input_shape = model.input_shape
    for exit_layer in model.exit_layers:
        # extract the conv layer from dynapcnn network
        og_readout_conv_layer = exit_layer.conv_layer
        og_weight_data = og_readout_conv_layer.weight.data
        og_bias_data = og_readout_conv_layer.bias
        og_bias = og_bias_data is not None
        # modify the out channels
        og_out_channels = og_readout_conv_layer.out_channels
        new_out_channels = (og_out_channels - 1) * 4 + 1
        og_readout_conv_layer.out_channels = new_out_channels
        # build extended weight and replace the old one
        ext_weight_shape = (new_out_channels, *og_weight_data.shape[1:])
        ext_weight_data = torch.zeros(ext_weight_shape, dtype=og_weight_data.dtype)
        for i in range(og_out_channels):
            ext_weight_data[i * 4] = og_weight_data[i]
        og_readout_conv_layer.weight.data = ext_weight_data
        # build extended bias and replace if necessary
        if og_bias:
            ext_bias_shape = (new_out_channels,)
            ext_bias_data = torch.zeros(ext_bias_shape, dtype=og_bias_data.dtype)
            for i in range(og_out_channels):
                ext_bias_data[i * 4] = og_bias_data[i]
            og_readout_conv_layer.bias.data = ext_bias_data
        # run a forward pass to initialize the new weights and last IAF
    model(torch.zeros(size=(1, *input_shape)))
    return model


def infer_input_shape(
    snn: nn.Module, input_shape: Optional[Tuple[int, int, int]] = None
) -> Tuple[int, int, int]:
    """Infer expected shape of input for `snn` either from `input_shape`
    or from `DVSLayer` instance within `snn` which provides it.

    If neither are available, raise an InputConfigurationError.
    If both are the case, verify that the information is consistent.

    Args:
        snn (nn.Module): The SNN whose input shape is to be inferred.
        input_shape (tuple or None): Explicitly provide input shape.
            If not None, must be of the format `(channels, height, width)`.

    Returns:
        The input shape to `snn`, in the format `(channels, height, width)`
    """
    if input_shape is not None and len(input_shape) != 3:
        raise InputConfigurationError(
            f"input_shape expected to have length 3 or None but input_shape={input_shape} given."
        )

    # Find `DVSLayer` instance and infer input shape from it
    input_shape_from_layer = None
    for module in snn.modules():
        if isinstance(module, DVSLayer):
            input_shape_from_layer = module.input_shape
            # Make sure `input_shape_from_layer` is identical to provided `input_shape`
            if input_shape is not None and input_shape != input_shape_from_layer:
                raise InputConfigurationError(
                    f"Input shape from `DVSLayer` {input_shape_from_layer} does "
                    f"not match the specified input_shape {input_shape}"
                )
            return input_shape_from_layer

    # If no `DVSLayer` is found, `input_shape` must not be provided
    if input_shape is None:
        raise InputConfigurationError(
            "No input shape could be inferred. Either provide it explicitly "
            "with the `input_shape` argument, or provide a model with "
            "`DVSLayer` instance."
        )
    else:
        return input_shape
