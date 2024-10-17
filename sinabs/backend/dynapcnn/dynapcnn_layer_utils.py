from math import prod
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from torch import nn

from sinabs import layers as sl

from .dynapcnn_layer import DynapcnnLayer
from .utils import expand_to_pair


def construct_dynapcnnlayers_from_mapper(
    dcnnl_map: Dict, discretize: bool, rescale_fn: Optional[Callable] = None
) -> Tuple[Dict[int, DynapcnnLayer], Dict[int, Set[int]], List[int]]:
    """Construct DynapcnnLayer instances from `dcnnl_map`

    Paramters
    ---------

    Returns
    -------
    - Dict of new DynapcnnLayer instances, with keys corresponding to `dcnnl_map`
    - Dict mapping to each layer index a set of destination indices
    - List of layer indices that act as entry points to the network
    """
    finalize_dcnnl_map(dcnnl_map, rescale_fn)

    dynapcnn_layers = {
        layer_idx: construct_single_dynapcnn_layer(layer_info, discretize)
        for layer_idx, layer_info in dcnnl_map.items()
    }

    destination_map = construct_destination_map(dcnnl_map)
    entry_points = collect_entry_points(dcnnl_map)

    return dynapcnn_layers, destination_map, entry_points


def finalize_dcnnl_map(dcnnl_map: Dict, rescale_fn: Optional[Callable] = None):
    """Finalize dcnnl map by consolidating information

    Update dcnnl_map in-place
    - Consolidate chained pooling layers
    - Determine rescaling of layer weights
    - Fix input shapes

    Parameters
    ----------
    - dcnnl_map: Dict holding info needed to instantiate DynapcnnLayer instances
    - rescale_fn: Optional callable that is used to determine layer
        rescaling in case of conflicting preceeding average pooling
    """
    # Consolidate pooling information for each destination
    for layer_info in dcnnl_map.values():
        consolidate_layer_pooling(layer_info, dcnnl_map)

    for layer_info in dcnnl_map.values():
        # Consolidate scale factors
        consolidate_layer_scaling(layer_info, rescale_fn)
        # Handle input dimensions
        determine_layer_input_shape(layer_info)


def consolidate_layer_pooling(layer_info: Dict, dcnnl_map: Dict):
    """Consolidate pooling information for individual layer

    Update `layer_info` and `dcnnl_map` in place.
    - Extract pooling and scale factor of consecutive pooling operations
    - To each "destination" add entries "cumulative_pooling" and
        "cumulative_scaling"
    - Add "pooling_list" to `layer_info` with all poolings of a layer
        in order of its "destination"s.
    - For each destination, add cumulative rescale factor to "rescale_factors"
        entry in corresponding entry of `dcnnl_map`.

    Parameters
    ----------
    - layer_info: Dict holding info of single layer. Corresponds to
        single entry in `dcnnl_map`
    - dcnnl_map: Dict holding info needed to instantiate DynapcnnLayer instances
    """
    layer_info["pooling_list"] = []
    for destination in layer_info["destinations"]:
        pool, scale = consolidate_dest_pooling(destination["pooling_modules"])
        destination["cumulative_pooling"] = pool
        layer_info["pooling_list"].append(pool)
        destination["cumulative_scaling"] = scale
        if (dest_lyr_idx := destination["destination_layer"]) is not None:
            dcnnl_map[dest_lyr_idx]["rescale_factors"].add(scale)


def consolidate_dest_pooling(
    modules: Iterable[nn.Module],
) -> Tuple[Tuple[int, int], float]:
    """Consolidate pooling information for consecutive pooling modules
    for single destination.

    Parameters
    ----------
    modules: Iteravle of pooling modules

    Returns
    -------
    cumulative_pooling: Tuple of two ints, indicating pooling along
        vertical and horizontal dimensions for all modules together
    cumulative_scaling: float, indicating by how much subsequent weights
        need to be rescaled to account for average pooling being converted
        to sum pooling, considering all provided modules.
    """
    cumulative_pooling = [1, 1]
    cumulative_scaling = 1.0

    for pooling_layer in modules:
        pooling, rescale_factor = extract_pooling_from_module(pooling_layer)
        cumulative_pooling[0] *= pooling[0]
        cumulative_pooling[1] *= pooling[1]
        cumulative_scaling *= rescale_factor

    return cumulative_pooling, cumulative_scaling


def extract_pooling_from_module(
    pooling_layer: Union[nn.AvgPool2d, sl.SumPool2d]
) -> Tuple[Tuple[int, int], float]:
    """Extract pooling size and required rescaling factor from pooling module

    Parameters
    ----------
    pooling_layer: pooling module

    Returns
    -------
    pooling: Tuple of two ints, indicating pooling along vertical and horizontal dimensions
    scale_factor: float, indicating by how much subsequent weights need to be rescaled to
        account for average pooling being converted to sum pooling.
    """
    pooling = expand_to_pair(pooling_layer.kernel_size)

    if pooling_layer.stride is not None:
        stride = expand_to_pair(pooling_layer.stride)
        if pooling != stride:
            raise ValueError(
                f"Stride length {pooling_layer.stride} should be the same as pooling kernel size {pooling_layer.kernel_size}"
            )
    if isinstance(pooling_layer, nn.AvgPool2d):
        scale_factor = 1.0 / (pooling[0] * pooling[1])
    elif isinstance(pooling_layer, sl.SumPool2d):
        scale_factor = 1.0
    else:
        raise ValueError(f"Unsupported type {type(pooling_layer)} for pooling layer")

    return pooling, scale_factor


def consolidate_layer_scaling(layer_info: Dict, rescale_fn: Optional[Callable] = None):
    """Dertermine scale factor of single layer

    Add "rescale_factor" entry to `layer_info`. If more than one
    different rescale factors have been determined due to conflicting
    average pooling in preceding layers, requrie `rescale_fn` to
    resolve.

    Parameters
    ----------
    - layer_info: Dict holding info of single layer.
    - rescale_fn: Optional callable that is used to determine layer
        rescaling in case of conflicting preceeding average pooling
    """
    if len(layer_info["rescale_factors"]) == 0:
        rescale_factor = 1
    elif len(layer_info["rescale_factors"]) == 1:
        rescale_factor = layer_info["rescale_factors"].pop()
    else:
        if rescale_fn is None:
            # TODO: Custom Exception class?
            raise ValueError(
                "Average pooling layers of conflicting sizes pointing to "
                "same destination. Either replace them by SumPool2d layers "
                "or provide a `rescale_fn` to resolve this"
            )
        else:
            rescale_factor = rescale_fn(layer_info["rescale_factors"])
    layer_info["rescale_factor"] = rescale_factor


def determine_layer_input_shape(layer_info: Dict):
    """Determine input shape of single layer

    Update "input_shape" entry of `layer_info`.
    If weight layer is convolutional, only verify that output shapes
    of preceding layer are not greater than input shape in any dimension.

    If weight layer is linear, the current "input_shape" entry will
    correspond to the shape after flattening, which might not match
    the shape of the actual input to the layer. Therefore the new input
    shape is the largest size across all output shapes of preceding
    layers, for each dimension individually.
    Verify that total number of elements (product of entries in new
    input shape) does not exceed that of original input shape.

    Parameters
    ----------
    - layer_info: Dict holding info of single layer.
    """
    # For each dimension find largest inferred input size
    max_inferred_input_shape = [
        max(sizes) for sizes in zip(*layer_info["inferred_input_shapes"])
    ]

    if isinstance(layer_info["conv"]["module"], nn.Linear):
        if prod(max_inferred_input_shape) > prod(layer_info["input_shape"]):
            raise ValueError(
                "Combined output of some layers projecting to a linear layer is "
                "larger than expected by destination layer. "
            )
        # Take shape before flattening, to convert linear to conv layer
        layer_info["input_shape"] = max_inferred_input_shape
    else:
        if any(
            inferred > expected
            for inferred, expected in zip(
                max_inferred_input_shape, layer_info["input_shape"]
            )
        ):
            raise ValueError(
                "Output of some layers is larger than expected by destination "
                "layer along some dimensions."
            )


def construct_single_dynapcnn_layer(
    layer_info: Dict, discretize: bool
) -> DynapcnnLayer:
    """Instantiate a DynapcnnLayer instance from the information
    in `layer_info'

    Parameters
    ----------
    - layer_info: Dict holding info of single layer.
    - discretize: bool indicating whether layer parameters should be
        discretized (weights, biases, thresholds)

    Returns
    -------
    """
    return DynapcnnLayer(
        conv=layer_info["conv"]["module"],
        spk=layer_info["neuron"]["module"],
        in_shape=layer_info["input_shape"],
        pool=layer_info["pooling_list"],
        discretize=discretize,
        rescale_weights=layer_info["rescale_factor"],
    )


def construct_destination_map(dcnnl_map: Dict[int, Dict]) -> Dict[int, List[int]]:
    """ Create a dict that holds destinations for each layer

    Parameters
    ----------
    - dcnnl_map: Dict holding info needed to instantiate DynapcnnLayer instances

    Returns
    -------
    Dict with layer indices (int) as keys and list of destination indices (int) as values.
        Layer outputs that are not sent to other dynapcnn layers are represented by negative indices.
    """
    destination_map = dict()
    for layer_index, layer_info in dcnnl_map.items():
        destination_indices = []
        none_counter = 0
        for dest in layer_info["destinations"]:
            if (dest_idx := dest["destination_layer"]) is None:
                # For `None` destinations use unique negative index
                none_counter += 1
                destination_indices.append(-none_counter)
            else:
                destination_indices.append(dest_idx)
        destination_map[layer_index] = destination_indices

    return destination_map


def collect_entry_points(dcnnl_map: Dict[int, Dict]) -> Set[int]:
    """ Return set of layer indices that are entry points

    Parameters
    ----------
    - dcnnl_map: Dict holding info needed to instantiate DynapcnnLayer instances

    Returns
    -------
    Set of all layer indices which act as entry points to the network
    """
    return {
        layer_index 
        for layer_index, layer_info in dcnnl_map.items() if layer_info["is_entry_node"]
    }
