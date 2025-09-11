from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

from torch import nn

from sinabs import layers as sl
from sinabs.utils import expand_to_pair

from .dynapcnn_layer import DynapcnnLayer


def construct_dynapcnnlayers_from_mapper(
    dcnnl_map: Dict,
    dvs_layer_info: Union[None, Dict],
    discretize: bool,
    rescale_fn: Optional[Callable] = None,
) -> Tuple[Dict[int, DynapcnnLayer], Dict[int, Set[int]], List[int]]:
    """Construct DynapcnnLayer instances from `dcnnl_map`

    Args:

    Returns:
     A tuple containing a dict of new DynapcnnLayer instances, with keys
    corresponding to `dcnnl_map`, a dict mapping each layer index to a set
    of destination indices and a list of layer indices that act as entry
    points to the network.
    """
    finalize_dcnnl_map(dcnnl_map, dvs_layer_info, rescale_fn)

    dynapcnn_layers = {
        layer_idx: construct_single_dynapcnn_layer(layer_info, discretize)
        for layer_idx, layer_info in dcnnl_map.items()
    }

    destination_map = construct_destination_map(dcnnl_map, dvs_layer_info)

    entry_points = collect_entry_points(dcnnl_map, dvs_layer_info)

    return dynapcnn_layers, destination_map, entry_points


def finalize_dcnnl_map(
    dcnnl_map: Dict, dvs_info: Union[Dict, None], rescale_fn: Optional[Callable] = None
) -> None:
    """Finalize DynapCNNLayer map by consolidating information

    Update `dcnnl_map` in-place
    - Consolidate chained pooling layers
    - Determine rescaling of layer weights
    - Fix input shapes

    Args:
        dcnnl_map: Dict holding info needed to instantiate DynapcnnLayer instances.
        rescale_fn: Optional callable that is used to determine layer rescaling
            in case of conflicting preceeding average pooling.
    """
    # Consolidate pooling information for DVS layer
    consolidate_dvs_pooling(dvs_info, dcnnl_map)

    # Consolidate pooling information for each destination
    for layer_info in dcnnl_map.values():
        consolidate_layer_pooling(layer_info, dcnnl_map)

    for layer_info in dcnnl_map.values():
        # Consolidate scale factors
        consolidate_layer_scaling(layer_info, rescale_fn)


def consolidate_dvs_pooling(dvs_info: Union[Dict, None], dcnnl_map: Dict):
    """Consolidate pooling information for DVS layer

    Update `dvs_info` and `dcnnl_map` in place.
    - Extract pooling and scale factor of consecutive pooling operations
    - Add entries "cumulative_pooling" and "cumulative_scaling"
    - Update DVSLayer pooling if applicable
    - For each destination, add cumulative rescale factor to "rescale_factors"
        entry in corresponding entry of `dcnnl_map`.

    Args:
        dvs_info: Dict holding info of DVS layer.
        dcnnl_map: Dict holding info needed to instantiate DynapcnnLayer instances.
    """
    if dvs_info is None or dvs_info["pooling"] is None:
        # Nothing to do
        return

    # Check whether pooling can be incorporated into the DVSLayer.
    dvs_layer = dvs_info["module"]
    crop_layer = dvs_layer.crop_layer
    if (
        crop_layer.top_crop != 0
        or crop_layer.left_crop != 0
        or crop_layer.bottom_crop != dvs_layer.input_shape[1]
        or crop_layer.right_crop != dvs_layer.input_shape[2]
    ):
        raise ValueError(
            "DVSLayer with cropping is followed by a pooling layer. "
            "This is currently not supported. Please define pooling "
            "directly within the DVSLayer (with the `pool` argument) "
            "and remove the pooling layer that follows the DVSLayer"
        )
    flip_layer = dvs_layer.flip_layer
    if flip_layer.flip_x or flip_layer.flip_y or flip_layer.swap_xy:
        raise ValueError(
            "DVSLayer with flipping or dimension swapping is followed "
            "by a pooling layer. This is currently not supported. "
            "Please define pooling directly within the DVSLayer "
            "(with the `pool` argument) and remove the pooling "
            "layer that follows the DVSLayer"
        )

    # Incorporate pooling into DVSLayer
    pool_layer = dvs_info["pooling"]["module"]
    added_pooling, scale = extract_pooling_from_module(pool_layer)
    dvs_pooling = expand_to_pair(dvs_layer.pool_layer.kernel_size)
    cumulative_pooling = (
        dvs_pooling[0] * added_pooling[0],
        dvs_pooling[1] * added_pooling[1],
    )
    dvs_layer.pool_layer.kernel_size = cumulative_pooling
    dvs_layer.pool_layer.stride = None

    # Update cropping layer to account for reduced size after pooling
    dvs_layer.crop_layer.bottom_crop //= added_pooling[0]
    dvs_layer.crop_layer.right_crop //= added_pooling[1]

    # Set rescale_factor for targeted dynapcnn layers
    if dvs_info["destinations"] is not None:
        for dest_lyr_idx in dvs_info["destinations"]:
            dcnnl_map[dest_lyr_idx]["rescale_factors"].add(scale)


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

    Args:
        layer_info: Dict holding info of single layer. Corresponds to single
            entry in `dcnnl_map`.
        dcnnl_map: Dict holding info needed to instantiate DynapcnnLayer instances.
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

    Args:
        modules: Iterable of pooling modules.

    Returns:
        A tuple containing the cumulative_pooling information, a tuple of two ints,
        indicating pooling along vertical and horizontal dimensions for all modules
        together and the cumulative_scaling, a float, indicating by how much subsequent
        weights need to be rescaled to account for average pooling being converted
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
    pooling_layer: Union[nn.AvgPool2d, sl.SumPool2d],
) -> Tuple[Tuple[int, int], float]:
    """Extract pooling size and required rescaling factor from pooling module

    Args:
        pooling_layer: pooling module.

    Returns:
        A tuple containing pooling, a tuple of two ints, indicating pooling along
        vertical and horizontal dimensions and the scale_factor, a float, indicating
        by how much subsequent weights need to be rescaled to account for average
        pooling being converted to sum pooling.
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
    """Determine scale factor of single layer

    Add "rescale_factor" entry to `layer_info`. If more than one
    different rescale factors have been determined due to conflicting
    average pooling in preceding layers, requires `rescale_fn` to
    resolve.

    Args:
        layer_info: Dict holding info of single layer.
        rescale_fn: Optional callable that is used to determine layer rescaling
            in case of conflicting preceeding average pooling.
    """
    if len(layer_info["rescale_factors"]) == 0:
        rescale_factor = 1
    elif len(layer_info["rescale_factors"]) == 1:
        rescale_factor = layer_info["rescale_factors"].pop()
    else:
        if rescale_fn is None:
            raise ValueError(
                "Average pooling layers of conflicting sizes pointing to "
                "same destination. Either replace them by SumPool2d layers "
                "or provide a `rescale_fn` to resolve this"
            )
        else:
            rescale_factor = rescale_fn(layer_info["rescale_factors"])
    layer_info["rescale_factor"] = rescale_factor


def construct_single_dynapcnn_layer(
    layer_info: Dict, discretize: bool
) -> DynapcnnLayer:
    """Instantiate a DynapcnnLayer instance from the information
    in `layer_info'

    Args:
        layer_info: Dict holding info of single layer.
        discretize: bool indicating whether layer parameters should be
            discretized (weights, biases, thresholds).

    Returns:
    """
    return DynapcnnLayer(
        conv=layer_info["conv"]["module"],
        spk=layer_info["neuron"]["module"],
        in_shape=layer_info["input_shape"],
        pool=layer_info["pooling_list"],
        discretize=discretize,
        rescale_weights=layer_info["rescale_factor"],
    )


def construct_destination_map(
    dcnnl_map: Dict[int, Dict], dvs_layer_info: Union[None, Dict]
) -> Dict[int, List[int]]:
    """Create a dict that holds destinations for each layer

    Args:
        dcnnl_map: Dict holding info needed to instantiate DynapcnnLayer instances.
        dynapcnn_layer_info: Dict holding info about DVSLayer instance and its destinations.

    Returns:
        Dict with layer indices (int) as keys and list of destination indices (int) as values.
        Layer outputs that are not sent to other dynapcnn layers are considered
        exit points of the network and represented by negative indices.
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
    if dvs_layer_info is not None:
        if (dest_info := dvs_layer_info["destinations"]) is None:
            destination_map["dvs"] = [-1]
        else:
            # Copy destination list from dvs layer info
            destination_map["dvs"] = [d for d in dest_info]

    return destination_map


def collect_entry_points(
    dcnnl_map: Dict[int, Dict], dvs_layer_info: Union[None, Dict]
) -> Set[int]:
    """Return set of layer indices that are entry points

    Args:
        dcnnl_map: Dict holding info needed to instantiate DynapcnnLayer instances.
        dvs_layer_info: Dict holding info about DVSLayer instance and its destinations.
            If it is not None, it will be the only entry point returned.

    Returns:
        Set of all layer indices which act as entry points to the network.
    """
    if dvs_layer_info is None:
        return {
            layer_index
            for layer_index, layer_info in dcnnl_map.items()
            if layer_info["is_entry_node"]
        }
    else:
        return {"dvs"}
