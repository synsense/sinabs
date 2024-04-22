from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union, Dict

import torch
import torch.nn as nn

import sinabs
import sinabs.layers as sl

from .crop2d import Crop2d
from .dvs_layer import DVSLayer, expand_to_pair
#from .dynapcnn_layer import DynapcnnLayer
from .dynapcnn_layer_new import DynapcnnLayer
from .exceptions import InputConfigurationError, MissingLayer, UnexpectedLayer, WrongModuleCount, WrongPoolingModule
from .flipdims import FlipDims

from .sinabs_edges_handler import process_edge, get_dynapcnnlayers_destinations

if TYPE_CHECKING:
    from sinabs.backend.dynapcnn.dynapcnn_network import DynapcnnNetwork

DEFAULT_IGNORED_LAYER_TYPES = (nn.Identity, nn.Dropout, nn.Dropout2d, nn.Flatten)


def infer_input_shape(
    layers: List[nn.Module], input_shape: Optional[Tuple[int, int, int]] = None
) -> Tuple[int, int, int]:
    """Checks if the input_shape is specified. If either of them are specified, then it checks if
    the information is consistent and returns the input shape.

    Parameters
    ----------
    layers:
        List of modules
    input_shape :
        (channels, height, width)

    Returns
    -------
    Output shape:
        (channels, height, width)
    """
    if input_shape is not None and len(input_shape) != 3:
        raise InputConfigurationError(
            f"input_shape expected to have length 3 or None but input_shape={input_shape} given."
        )

    input_shape_from_layer = None
    if layers and isinstance(layers[0], DVSLayer):
        input_shape_from_layer = layers[0].input_shape
        if len(input_shape_from_layer) != 3:
            raise InputConfigurationError(
                f"input_shape of layer {layers[0]} expected to have length 3 or None but input_shape={input_shape_from_layer} found."
            )
    if (input_shape is not None) and (input_shape_from_layer is not None):
        if input_shape == input_shape_from_layer:
            return input_shape
        else:
            raise InputConfigurationError(
                f"Input shape from the layer {input_shape_from_layer} does not match the specified input_shape {input_shape}"
            )
    elif input_shape_from_layer is not None:
        return input_shape_from_layer
    elif input_shape is not None:
        return input_shape
    else:
        raise InputConfigurationError("No input shape could be inferred")


def convert_cropping2dlayer_to_crop2d(
    layer: sl.Cropping2dLayer, input_shape: Tuple[int, int]
) -> Crop2d:
    """Convert a sinabs layer of type Cropping2dLayer to Crop2d layer.

    Parameters
    ----------
    layer:
        Cropping2dLayer
    input_shape:
        (height, width) input dimensions

    Returns
    -------
    Equivalent Crop2d layer
    """
    h, w = input_shape
    top = layer.top_crop
    left = layer.left_crop
    bottom = h - layer.bottom_crop
    right = w - layer.right_crop
    print(h, w, left, right, top, bottom, layer.right_crop, layer.bottom_crop)
    return Crop2d(((top, bottom), (left, right)))


def construct_dvs_layer(
    layers: List[nn.Module],
    input_shape: Tuple[int, int, int],
    idx_start: int = 0,
    dvs_input: bool = False,
) -> Tuple[Optional[DVSLayer], int, float]:
    """
    Generate a DVSLayer given a list of layers. If `layers` does not start
    with a pooling, cropping or flipping layer and `dvs_input` is False,
    will return `None` instead of a DVSLayer.
    NOTE: The number of channels is implicitly assumed to be 2 because of DVS

    Parameters
    ----------
    layers:
        List of layers
    input_shape:
        Shape of input (channels, height, width)
    idx_start:
        Starting index to scan the list. Default 0

    Returns
    -------
    dvs_layer:
        None or DVSLayer
    idx_next: int or None
        Index of first layer after this layer is constructed
    rescale_factor: float
        Rescaling factor needed when turning AvgPool to SumPool. May
        differ from the pooling kernel in certain cases.
    dvs_input: bool
        Whether DVSLayer should have pixel array activated.
    """
    # Start with defaults
    layer_idx_next = idx_start
    crop_lyr = None
    flip_lyr = None

    if len(input_shape) != 3:
        raise ValueError(
            f"Input shape should be 3 dimensional but input_shape={input_shape} was given."
        )

    # Return existing DVS layer as is
    if len(layers) and isinstance(layers[0], DVSLayer):
        return deepcopy(layers[0]), 1, 1

    # Construct pooling layer
    pool_lyr, layer_idx_next, rescale_factor = construct_next_pooling_layer(
        layers, layer_idx_next
    )

    # Find next layer (check twice for two layers)
    for __ in range(2):
        # Go to the next layer
        if layer_idx_next < len(layers):
            layer = layers[layer_idx_next]
        else:
            break
        # Check layer type
        if isinstance(layer, sl.Cropping2dLayer):
            # The shape after pooling is
            pool = expand_to_pair(pool_lyr.kernel_size)
            h = input_shape[1] // pool[0]
            w = input_shape[2] // pool[1]
            print(f"Input shape to the cropping layer is {h}, {w}")
            crop_lyr = convert_cropping2dlayer_to_crop2d(layer, (h, w))
        elif isinstance(layer, Crop2d):
            crop_lyr = layer
        elif isinstance(layer, FlipDims):
            flip_lyr = layer
        else:
            break

        layer_idx_next += 1

    # If any parameters have been found or dvs_input is True
    if (layer_idx_next > 0) or dvs_input:
        dvs_layer = DVSLayer.from_layers(
            pool_layer=pool_lyr,
            crop_layer=crop_lyr,
            flip_layer=flip_lyr,
            input_shape=input_shape,
            disable_pixel_array=not dvs_input,
        )
        return dvs_layer, layer_idx_next, rescale_factor
    else:
        # No parameters/layers pertaining to DVS preprocessing found
        return None, 0, 1


def merge_conv_bn(conv, bn):
    """Merge a convolutional layer with subsequent batch normalization.

    Parameters
    ----------
        conv: torch.nn.Conv2d
            Convolutional layer
        bn: torch.nn.Batchnorm2d
            Batch normalization

    Returns
    -------
        torch.nn.Conv2d: Convolutional layer including batch normalization
    """
    mu = bn.running_mean
    sigmasq = bn.running_var

    if bn.affine:
        gamma, beta = bn.weight, bn.bias
    else:
        gamma, beta = 1.0, 0.0

    factor = gamma / sigmasq.sqrt()

    c_weight = conv.weight.data.clone().detach()
    c_bias = 0.0 if conv.bias is None else conv.bias.data.clone().detach()

    conv = deepcopy(conv)  # TODO: this will cause copying twice

    conv.weight.data = c_weight * factor[:, None, None, None]
    conv.bias.data = beta + (c_bias - mu) * factor

    return conv


def construct_next_pooling_layer(
    layers: List[nn.Module], idx_start: int
) -> Tuple[Optional[sl.SumPool2d], int, float]:
    """Consolidate the first `AvgPool2d` objects in `layers` until the first object of different
    type.

    Parameters
    ----------
        layers: Sequence of layer objects
            Contains `AvgPool2d` and other objects.
        idx_start: int
            Layer index to start construction from
    Returns
    -------
        lyr_pool: int or tuple of ints
            Consolidated pooling size.
        idx_next: int
            Index of first object in `layers` that is not a `AvgPool2d`,
        rescale_factor: float
            Rescaling factor needed when turning AvgPool to SumPool. May
            differ from the pooling kernel in certain cases.
    """

    rescale_factor = 1
    cumulative_pooling = expand_to_pair(1)
    idx_next = idx_start
    # Figure out pooling dims
    while idx_next < len(layers):
        lyr = layers[idx_next]
        if isinstance(lyr, nn.AvgPool2d):
            if lyr.padding != 0:
                raise ValueError("Padding is not supported for the pooling layers")
        elif isinstance(lyr, sl.SumPool2d):
            ...
        else:
            # Reached a non pooling layer
            break
        # Increment if it is a pooling layer
        idx_next += 1

        pooling = expand_to_pair(lyr.kernel_size)
        if lyr.stride is not None:
            stride = expand_to_pair(lyr.stride)
            if pooling != stride:
                raise ValueError(
                    f"Stride length {lyr.stride} should be the same as pooling kernel size {lyr.kernel_size}"
                )
        # Compute cumulative pooling
        cumulative_pooling = (
            cumulative_pooling[0] * pooling[0],
            cumulative_pooling[1] * pooling[1],
        )
        
        # Update rescaling factor
        if isinstance(lyr, nn.AvgPool2d):
            rescale_factor *= pooling[0] * pooling[1]

    # If there are no layers
    if cumulative_pooling == (1, 1):
        return None, idx_next, 1
    else:
        lyr_pool = sl.SumPool2d(cumulative_pooling)
        return lyr_pool, idx_next, rescale_factor


def construct_next_dynapcnn_layer(
    layers: List[nn.Module],
    idx_start: int,
    in_shape: Tuple[int, int, int],
    discretize: bool,
    rescale_factor: float = 1,
) -> Tuple[DynapcnnLayer, int, float]:
    """Generate a DynapcnnLayer from a Conv2d layer and its subsequent spiking and pooling layers.

    Parameters
    ----------

        layers: sequence of layer objects
            First object must be Conv2d, next must be an IAF layer. All pooling
            layers that follow immediately are consolidated. Layers after this
            will be ignored.
        idx_start:
            Layer index to start construction from
        in_shape: tuple of integers
            Shape of the input to the first layer in `layers`. Convention:
            (input features, height, width)
        discretize: bool
            Discretize weights and thresholds if True
        rescale_factor: float
            Weights of Conv2d layer are scaled down by this factor. Can be
            used to account for preceding average pooling that gets converted
            to sum pooling.

    Returns
    -------
        dynapcnn_layer: DynapcnnLayer
            DynapcnnLayer
        layer_idx_next: int
            Index of the next layer after this layer is constructed
        rescale_factor: float
            rescaling factor to account for average pooling
    """
    layer_idx_next = idx_start  # Keep track of layer indices

    # Check that the first layer is Conv2d, or Linear
    if not isinstance(layers[layer_idx_next], (nn.Conv2d, nn.Linear)):
        raise UnexpectedLayer(nn.Conv2d, layers[layer_idx_next])

    # Identify and consolidate conv layer
    lyr_conv = layers[layer_idx_next]
    layer_idx_next += 1
    if layer_idx_next >= len(layers):
        raise MissingLayer(layer_idx_next)
    # Check and consolidate batch norm
    if isinstance(layers[layer_idx_next], nn.BatchNorm2d):
        lyr_conv = merge_conv_bn(lyr_conv, layers[layer_idx_next])
        layer_idx_next += 1

    # Check next layer exists
    try:
        lyr_spk = layers[layer_idx_next]
        layer_idx_next += 1
    except IndexError:
        raise MissingLayer(layer_idx_next)

    # Check that the next layer is spiking
    # TODO: Check that the next layer is an IAF layer
    if not isinstance(lyr_spk, sl.IAF):
        raise TypeError(
            f"Convolution must be followed by IAF spiking layer, found {type(lyr_spk)}"
        )

    # Check for next pooling layer
    lyr_pool, i_next, rescale_factor_after_pooling = construct_next_pooling_layer(
        layers, layer_idx_next
    )
    # Increment layer index to after the pooling layers
    layer_idx_next = i_next

    # Compose DynapcnnLayer
    dynapcnn_layer = DynapcnnLayer(
        conv=lyr_conv,
        spk=lyr_spk,
        pool=lyr_pool,
        in_shape=in_shape,
        discretize=discretize,
        rescale_weights=rescale_factor,
    )

    return dynapcnn_layer, layer_idx_next, rescale_factor_after_pooling


def build_from_list(
    layers: List[nn.Module],
    in_shape,
    discretize=True,
    dvs_input=False,
) -> nn.Sequential:
    """Build a sequential model of DVSLayer and DynapcnnLayer(s) given a list of layers comprising
    a spiking CNN.

    Parameters
    ----------

        layers: sequence of layer objects
        in_shape: tuple of integers
            Shape of the input to the first layer in `layers`. Convention:
            (channels, height, width)
        discretize: bool
            Discretize weights and thresholds if True
        dvs_input: bool
            Whether model should receive DVS input. If `True`, the returned model
            will begin with a DVSLayer with `disable_pixel_array` set to False.
            Otherwise, the model starts with a DVSLayer only if the first element
            in `layers` is a pooling, cropping or flipping layer.

    Returns
    -------
        nn.Sequential
    """
    compatible_layers = []
    lyr_indx_next = 0
    # Find and populate dvs layer (NOTE: We are ignoring the channel information here and could lead to problems)
    dvs_layer, lyr_indx_next, rescale_factor = construct_dvs_layer(
        layers, input_shape=in_shape, idx_start=lyr_indx_next, dvs_input=dvs_input
    )
    
    if dvs_layer is not None:
        compatible_layers.append(dvs_layer)
        in_shape = dvs_layer.get_output_shape()
    # Find and populate dynapcnn layers
    while lyr_indx_next < len(layers):
        if isinstance(layers[lyr_indx_next], DEFAULT_IGNORED_LAYER_TYPES):
            # - Ignore identity, dropout and flatten layers
            lyr_indx_next += 1
            continue
        dynapcnn_layer, lyr_indx_next, rescale_factor = construct_next_dynapcnn_layer(
            layers,
            lyr_indx_next,
            in_shape=in_shape,
            discretize=discretize,
            rescale_factor=rescale_factor,
        )
        in_shape = dynapcnn_layer.get_output_shape()
        compatible_layers.append(dynapcnn_layer)

    return nn.Sequential(*compatible_layers)


def convert_model_to_layer_list(
    model: Union[nn.Sequential, sinabs.Network, nn.Module],
    ignore: Union[Type, Tuple[Type, ...]] = (),
) -> List[nn.Module]:
    """Convert a model to a list of layers.

    Parameters
    ----------
        model: nn.Sequential, nn.Module or sinabs.Network.
        ignore: type or tuple of types of modules to be ignored.

    Returns
    -------
        List[nn.Module]
    """
    if isinstance(model, sinabs.Network):
        return convert_model_to_layer_list(model.spiking_model)
    
    elif isinstance(model, nn.Sequential):
        layers = [layer for layer in model if not isinstance(layer, ignore)]

    elif isinstance(model, nn.Module):
        layers = [layer for _, layer in model.named_children() if not isinstance(layer, ignore)]

    else:
        raise TypeError("Expected torch.nn.Sequential or sinabs.Network")
    
    return layers


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
    og_readout_conv_layer = model.sequence[
        -1
    ].conv_layer  # extract the conv layer from dynapcnn network
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
    _ = model(
        torch.zeros(size=(1, *input_shape))
    )  # run a forward pass to initialize the new weights and last IAF
    return model

def build_nodes_to_dcnnl_map(layers: Dict[int, nn.Module], edges: List[Tuple[int, int]]):
    """ ."""
    # @TODO the graph extraction is not yet considering DVS input.

    # dvs_layer, lyr_indx_next, rescale_factor = construct_dvs_layer(
    #     layers, 
    #     input_shape=in_shape, 
    #     idx_start=0, 
    #     dvs_input=False)
    
    dvs_layer = None

    nodes_to_dcnnl_map = {}                     # mapper from nodes to sets of layers that populate a DynapcnnLayer.

    if dvs_layer is not None:
        pass                                    # TODO the graph extraction is not yet considering DVS input.
    else:
        for edge in edges:
            process_edge(                       # figure out to which (future) DynapcnnLayer each node will belong to.
                layers, edge, nodes_to_dcnnl_map)

    get_dynapcnnlayers_destinations(            # look for edges between connecting nodes in different (future) DynapcnnLayer.
        layers, edges, nodes_to_dcnnl_map)
    
    return nodes_to_dcnnl_map

def build_from_graph(
        discretize: bool,
        edges: List[Tuple[int, int]],
        nodes_to_dcnnl_map: dict) -> dict:
    """ Parses each edge of a 'sinabs_mode.spiking_model' computational graph. Each node (layer) is assigned to a 
    DynapcnnLayer object. The target destination of each DynapcnnLayer is computed via edges connecting nodes in 
    different DynapcnnLayer objects.

    Parameters
    ----------
        ...

    Returns
    ----------
        ...
    """


    dynapcnn_layers = construct_dynapcnnlayers_from_mapper(                     # turn sets of layers into DynapcnnLayer objects.
        discretize, nodes_to_dcnnl_map, edges)
    
    for idx, layer_data in dynapcnn_layers.items():
        if 'core_idx' not in layer_data:
            layer_data['core_idx'] = -1         # a DynapcnnLayer gets assigned a core index when 'DynapcnnNetworkGraph.to()' is called.
    
    return dynapcnn_layers

def construct_dynapcnnlayers_from_mapper(
        discretize: bool,
        nodes_to_dcnnl_map: dict,
        edges: List[Tuple[int, int]]) -> Dict[int, Dict[DynapcnnLayer, List]]:
    """ Consumes a dictionaries containing sets of layers to be used to populate a DynapcnnLayer object.

    Parameters
    ----------
        rescale_factor: Rescaling factor needed when turning AvgPool to SumPool. May differ from the pooling kernel in 
                        certain cases.
    Returns
    ----------
        dynapcnn_layers: A dictionary containing DynapcnnLayer objects and a list with their destinations.
    """

    dynapcnn_layers = {}
    
    for dpcnnl_idx, dcnnl_data in nodes_to_dcnnl_map.items():
        dynapcnnlayer = construct_dynapcnnlayer(
            discretize, dcnnl_data, edges, nodes_to_dcnnl_map)
        
        dynapcnn_layers[dpcnnl_idx] = {
            'layer': dynapcnnlayer, 
            'destinations': nodes_to_dcnnl_map[dpcnnl_idx]['destinations']
            }

        node, output_shape = dynapcnnlayer.get_modified_node_it(dcnnl_data)

        if isinstance(node, int) and isinstance(output_shape, tuple):
            update_nodes_io(node, output_shape, nodes_to_dcnnl_map, edges)

    return dynapcnn_layers

def update_nodes_io(updated_node: int, output_shape: tuple, nodes_to_dcnnl_map: dict, edges: List[Tuple[int, int]]) -> None:
    """ ."""
    for edge in edges:
        if edge[0] == updated_node:
            # found source node where output shape has been modified.
            for dcnnl_idx, dcnnl_data in nodes_to_dcnnl_map.items():
                for key, val in dcnnl_data.items():
                    if isinstance(key, int):
                        # accessing node data (layer, input_shape, output_shape).
                        if key == edge[1]:
                            # accessing node targeted by `updated_node` (its input shape becomes `updated_node.output_shape`).
                            val['input_shape'] = output_shape

def construct_dynapcnnlayer(
        discretize: bool,
        dcnnl_data: dict, 
        edges: List[Tuple[int, int]],
        nodes_to_dcnnl_map: dict) -> DynapcnnLayer:
    """ Extract the modules (layers) in a dictionary and uses them to instantiate a DynapcnnLayer object. 

    Parameters
    ----------
        dcnnl_data: contains the nodes to be merged into a DynapcnnLayer, their I/O shapes and the index of the other DynapcnnLayers to
                    be set as destinations.
    """

    # convert all AvgPool2d in 'dcnnl_data' into SumPool2d.
    convert_Avg_to_Sum_pooling(dcnnl_data, edges, nodes_to_dcnnl_map)

    # TODO 'rescale_weight' information is inside 'dcnnl_data' but it is not yet being used.
    # TODO the input shapes work fine when processing conv layers but once we reach linear layers
    # their input shapes has to come from the conv layers projecting to the node. Currently also a 
    # linear layer after a flatten has the input shape coming out of the flatten but it needs to be
    # the conv output shape before the flatten. A linear layer coming after a previous linear layer 
    # converted into conv needs to have its input shape updated based on this conversion.

    # instantiate a DynapcnnLayer from the data in 'dcnnl_data'.
    dynapcnnlayer = DynapcnnLayer(
        dcnnl_data      = dcnnl_data,
        discretize      = discretize,
        nodes_mapper    = nodes_to_dcnnl_map
    )

    return dynapcnnlayer

def convert_Avg_to_Sum_pooling(dcnnl_data: dict, edges: list, nodes_to_dcnnl_map: dict):
    """ Converts every `AvgPool2d` node within `dcnnl_data` into a `SumPool2d` and update their respective `rescale_factor` (to
    be used when creating the `DynapcnnLayer` instance for this layer's destinations).

    Parameters
    ----------
        dcnnl_data: ...
        edges: ...
        nodes_to_dcnnl_map: ...
    """
    for key, value in dcnnl_data.items():
        if isinstance(key, int):
            # accessing the node 'key's dictionary.

            if isinstance(value['layer'], nn.AvgPool2d):
                # convert AvgPool2d into SumPool2d.
                lyr_pool, rescale_factor = build_SumPool2d(value['layer'])

                # turn avg into sum pool.
                value['layer'] = lyr_pool

                # find which node 'key' will target.
                for edge in edges:
                    if edge[0] == key:
                        # target within DynapcnnLayer index 'trg_dcnnl_idx'.
                        trg_dcnnl_idx = find_nodes_dcnnl_idx(edge[1], nodes_to_dcnnl_map)

                        # update the rescale factor for the target of node 'key'.
                        dcnnl_data['destinations_rescale_factor'][trg_dcnnl_idx] = rescale_factor

def find_nodes_dcnnl_idx(node, nodes_to_dcnnl_map):
    """ ."""
    for dcnnl_idx, dcnnl_data in nodes_to_dcnnl_map.items():
        for key, value in dcnnl_data.items():
            if isinstance(key, int):
                # 'key' is a node.
                if key == node:
                    # node belongs to DynapcnnLayer index 'dcnnl_idx'.
                    return dcnnl_idx

    # this exception should never happen.
    raise ValueError(f'Node {node} is not part of any dictionary mapping into a DynapcnnLayer.')

def build_SumPool2d(module: nn.AvgPool2d) -> Tuple[sl.SumPool2d, int]:
    """ Converts a 'nn.AvgPool2d' into a 'sl.SumPool2d' layer. """
    
    if isinstance(module, nn.AvgPool2d):
        if module.padding != 0:
            raise ValueError("Padding is not supported for the pooling layers.")
    elif isinstance(module, sl.SumPool2d):
        pass
    else:
        raise WrongPoolingModule(type(module))
    
    rescale_factor = 1
    cumulative_pooling = expand_to_pair(1)
    pooling = expand_to_pair(module.kernel_size)

    if module.stride is not None:
        stride = expand_to_pair(module.stride)
        if pooling != stride:
            raise ValueError(
                f"Stride length {module.stride} should be the same as pooling kernel size {module.kernel_size}"
            )

    cumulative_pooling = (                  # compute cumulative pooling.
        cumulative_pooling[0] * pooling[0],
        cumulative_pooling[1] * pooling[1],
    )

    if isinstance(module, nn.AvgPool2d):    # update rescaling factor.
        rescale_factor *= pooling[0] * pooling[1]

    lyr_pool = sl.SumPool2d(cumulative_pooling)

    return lyr_pool, rescale_factor