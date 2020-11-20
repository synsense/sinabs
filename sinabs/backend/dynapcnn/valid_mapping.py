from math import ceil, log2
from typing import List, TypeVar

DynapcnnConfiguration = TypeVar("DynapcnnConfiguration")
CNNLayerConfig = TypeVar("CNNLayerConfig")

_WEIGHTS_MEMORY_SIZE = [
    16 * 1024,  # 0
    16 * 1024,  # 1
    16 * 1024,  # 2
    32 * 1024,  # 3
    32 * 1024,  # 4
    64 * 1024,  # 5
    64 * 1024,  # 6
    16 * 1024,  # 7
    16 * 1024,
]  # _WEIGHTS_MEMORY_SIZE

_NEURONS_MEMORY_SIZE = [
    64 * 1024,  # 0
    64 * 1024,  # 1
    64 * 1024,  # 2
    32 * 1024,  # 3
    32 * 1024,  # 4
    16 * 1024,  # 5
    16 * 1024,  # 6
    16 * 1024,  # 7
    16 * 1024,
]  # 8


def _get_output_size(
    input_feature_size: int, kernel_size: int, padding: int, stride: int
) -> int:
    """Output size for a given input size and layer dimensions along one axis"""
    return (input_feature_size - kernel_size + 2 * padding) / stride + 1


def _min_bits_required(value: int) -> int:
    """Minimum number of bits required to represent a given value"""
    assert value != 0
    return ceil(log2(value))


def _compute_weight_memory(config: CNNLayerConfig) -> int:
    """Required memory size to store CNN layer weights"""
    power = _min_bits_required(
        config.dimensions.output_shape.feature_count
    ) + _min_bits_required(
        config.dimensions.kernel_size * config.dimensions.kernel_size
    )
    return config.dimensions.input_shape.feature_count * (1 << power)


def _compute_neuron_memory(config: CNNLayerConfig) -> int:
    """Required memory size to store CNN layer neuron states"""
    fx = _get_output_size(
        config.dimensions.input_shape.size.x,
        config.dimensions.kernel_size,
        config.dimensions.padding.x,
        config.dimensions.stride.x,
    )

    fy = _get_output_size(
        config.dimensions.input_shape.size.y,
        config.dimensions.kernel_size,
        config.dimensions.padding.y,
        config.dimensions.stride.y,
    )

    power = _min_bits_required(fx) + _min_bits_required(fy)
    return config.dimensions.output_shape.feature_count * (1 << power)


def get_valid_mapping(config: DynapcnnConfiguration) -> List[List[int]]:
    """Find valid remapping of layers in DYNAPCNN config

    Returns a valid remapping of the layers in dynapcnn config if it finds one.
    The returned value is a list of indexes from the current config
    how they should be mapped in order to fit the memory.

    Parameters
    ----------
        config: samna.dynapcnn.configuration.DynapcnnConfiguration
            The DYNAPCNN configuration whose mapping should be validated

    Returns
    -------
        List[List[int]]
            List of index pairs (i, j) indicating that the i-th layer in `config`
            should be mapped to the j-th layer on DYNAPCNN

    """

    mapping = []

    memory_values = []
    memory_limits = []

    # find all layers used as destination
    # for this we check for DVS and all layers destination enable flag

    used_layers = []
    # finds layers that are targets of the DVS input
    for destination in config.dvs_layer.destinations:
        if destination.enable and not (destination.layer in used_layers):
            used_layers.append(destination.layer)

    # finds layers that are targets of any other layer
    for selected_layer in range(0, len(config.cnn_layers)):
        for destination in config.cnn_layers[selected_layer].destinations:
            if destination.enable and not (destination.layer in used_layers):
                used_layers.append(destination.layer)

    # find all layers that have a target
    for selected_layer in range(0, len(config.cnn_layers)):
        for destination in config.cnn_layers[selected_layer].destinations:
            if destination.enable and not (selected_layer in used_layers):
                used_layers.append(selected_layer)

    for selected_layer in used_layers:
        weight_memory = _compute_weight_memory(config.cnn_layers[selected_layer])
        neuron_memory = _compute_neuron_memory(config.cnn_layers[selected_layer])
        memory_values.append([selected_layer, [weight_memory, neuron_memory]])

    for selected_layer in range(0, len(_WEIGHTS_MEMORY_SIZE)):
        memory_limits.append(
            [
                selected_layer,
                [
                    _WEIGHTS_MEMORY_SIZE[selected_layer],
                    _NEURONS_MEMORY_SIZE[selected_layer],
                ],
            ]
        )

    memory_values = sorted(memory_values, key=lambda x: (x[1][0], x[1][1]))
    memory_limits = sorted(memory_limits, key=lambda x: (x[1][0], x[1][1]))

    memory_values_index = len(memory_values) - 1
    memory_limits_index = len(memory_limits) - 1
    total_swaps = 0
    while memory_values_index >= 0:
        if (
            memory_values[memory_values_index][1][0]
            <= memory_limits[memory_limits_index][1][0]
            and memory_values[memory_values_index][1][1]
            <= memory_limits[memory_limits_index][1][1]
        ):
            mapping.append(
                [
                    memory_values[memory_values_index][0],
                    memory_limits[memory_limits_index][0],
                ]
            )
            # print(mapping)
            memory_values_index = memory_values_index - 1
            memory_limits_index = memory_limits_index - 1
        else:
            to_be_swapped_index = memory_values_index
            swapped = False
            while memory_values_index < len(memory_values) - 1:
                mapping = mapping[0 : len(mapping) - 2]
                if (
                    memory_values[memory_values_index][1][0]
                    < memory_values[memory_values_index + 1][1][0]
                    and memory_values[memory_values_index][1][1]
                    > memory_values[memory_values_index + 1][1][1]
                ):
                    # print("swapping " + str(memory_values[to_be_swapped_index][0]) + " " + str(memory_values[to_be_swapped_index][1][0]) + " " + str(memory_values[to_be_swapped_index][1][1]) + " with " + str(memory_values[memory_values_index+1][0]) + " " + str(memory_values[memory_values_index+1][1][0]) + " " + str(memory_values[memory_values_index+1][1][1]))
                    layer = memory_values[to_be_swapped_index][0]
                    weight = memory_values[to_be_swapped_index][1][0]
                    neuron = memory_values[to_be_swapped_index][1][1]
                    memory_values[to_be_swapped_index][0] = memory_values[
                        memory_values_index + 1
                    ][0]
                    memory_values[to_be_swapped_index][1][0] = memory_values[
                        memory_values_index + 1
                    ][1][0]
                    memory_values[to_be_swapped_index][1][1] = memory_values[
                        memory_values_index + 1
                    ][1][1]
                    memory_values[memory_values_index + 1][0] = layer
                    memory_values[memory_values_index + 1][1][0] = weight
                    memory_values[memory_values_index + 1][1][1] = neuron
                    mapping = []
                    memory_values_index = len(memory_values) - 1
                    memory_limits_index = len(memory_limits) - 1
                    swapped = True
                    total_swaps = total_swaps + 1
                    if total_swaps > 9:
                        print("can't find a solution!")
                        return []
                    break
                else:
                    memory_values_index = memory_values_index + 1
            if not swapped:
                print(
                    str(memory_values[to_be_swapped_index])
                    + " can't be mapped because it is too big! limit:"
                    + str(memory_limits[memory_limits_index])
                )
                return []

    return mapping
