from math import ceil, log2

weightsMemorySize = [
    16 * 1024,  # 0
    16 * 1024,  # 1
    16 * 1024,  # 2
    32 * 1024,  # 3
    32 * 1024,  # 4
    64 * 1024,  # 5
    64 * 1024,  # 6
    16 * 1024,  # 7
    16 * 1024]  # 8

neuronsMemorySize = [
    64 * 1024,  # 0
    64 * 1024,  # 1
    64 * 1024,  # 2
    32 * 1024,  # 3
    32 * 1024,  # 4
    16 * 1024,  # 5
    16 * 1024,  # 6
    16 * 1024,  # 7
    16 * 1024]  # 8


def getOutputSize(inputFeatureSize, kernelSize, padding, stride):
    return (inputFeatureSize - kernelSize + 2 * padding) / stride + 1


def minBitsRequired(value):
    assert(value != 0)
    return ceil(log2(value))


def computeWeightMemory(config):
    power = minBitsRequired(config.dimensions.output_shape.feature_count) + \
        minBitsRequired(config.dimensions.kernel_size * config.dimensions.kernel_size)
    return config.dimensions.input_shape.feature_count * (1 << power)


def computeNeuronMemory(config):
    fx = getOutputSize(
        config.dimensions.input_shape.size.x,
        config.dimensions.kernel_size,
        config.dimensions.padding.x,
        config.dimensions.stride.x)

    fy = getOutputSize(
        config.dimensions.input_shape.size.y,
        config.dimensions.kernel_size,
        config.dimensions.padding.y,
        config.dimensions.stride.y)

    power = minBitsRequired(fx) + minBitsRequired(fy)
    return config.dimensions.output_shape.feature_count * (1 << power)


def get_valid_mapping(config):
    """
    returns a valid remapping of the layers in speck config if it finds one
    The returned value is a list of indexes from the current config
    how they should be mapped in order to fit the memory.

    :param configuration: speck configuration
    :return: mapping -- a list of indexes
    """
    assert False

    mapping = []

    memoryValues = []
    memoryLimits = []

    # find all layers used as destination
    # for this we check for DVS and all layers destination enable flag

    usedLayers = []
    for destination in config.dvs_layer.destinations:
        if destination.enable and not (destination.layer in usedLayers):
            usedLayers.append(destination.layer)

    for selectedLayer in range(0, len(config.cnn_layers)):
        for destination in config.cnn_layers[selectedLayer].destinations:
            if destination.enable and not (destination.layer in usedLayers):
                usedLayers.append(destination.layer)

    for selectedLayer in usedLayers:
        weightMemory = computeWeightMemory(config.cnn_layers[selectedLayer])
        neuronMemory = computeNeuronMemory(config.cnn_layers[selectedLayer])
        memoryValues.append([selectedLayer, [weightMemory, neuronMemory]])

    for selectedLayer in range(0, len(weightsMemorySize)):
        memoryLimits.append([selectedLayer, [weightsMemorySize[selectedLayer], neuronsMemorySize[selectedLayer]]])

    memoryValues = sorted(memoryValues, key=lambda x: (x[1][0], x[1][1]))
    memoryLimits = sorted(memoryLimits, key=lambda x: (x[1][0], x[1][1]))

    memoryValuesIndex = len(memoryValues) - 1
    memoryLimitsIndex = len(memoryLimits) - 1
    while memoryValuesIndex >= 0:
        if memoryValues[memoryValuesIndex][1][0] <= memoryLimits[memoryLimitsIndex][1][0] and memoryValues[memoryValuesIndex][1][1] <= memoryLimits[memoryLimitsIndex][1][1]:
            mapping.append([memoryValues[memoryValuesIndex][0], memoryLimits[memoryLimitsIndex][0]])
            memoryValuesIndex = memoryValuesIndex - 1
            memoryLimitsIndex = memoryLimitsIndex - 1
        else:
            print(str(memoryValues[memoryValuesIndex]) + " can't be mapped because it is too big! limit:" + str(memoryLimits[memoryLimitsIndex]))
            memoryValuesIndex = memoryValuesIndex - 1

        selectedLayer = selectedLayer - 1

    return mapping
