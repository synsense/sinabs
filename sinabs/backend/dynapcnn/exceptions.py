from typing import Set, Tuple, Type


class MissingLayer(Exception):
    index: int

    def __init__(self, index: int):
        super().__init__(f"A spiking layer is expected at index {index}")


class UnexpectedLayer(Exception):
    layer_type_found: Type
    layer_type_expected: Type

    def __init__(self, expected, found):
        super().__init__(f"Expected {expected} but found {found}")


class InputConfigurationError(Exception):
    """Is raised when input to dynapcnn is not configured correctly."""

    pass


class WrongModuleCount(Exception):
    dynapcnnlayer_indx: Type
    modules_count: Type

    def __init__(self, dynapcnnlayer_indx, modules_count):
        super().__init__(
            f"A DynapcnnLayer {dynapcnnlayer_indx} should have 2 or 3 modules but found {modules_count}."
        )


class WrongPoolingModule(Exception):
    pooling_module: Type

    def __init__(
        self,
        pooling_module,
    ):
        super().__init__(
            f"The function 'utils.build_SumPool2d(mod)' expects 'mod = nn.AvgPool2d' but got 'mod = {pooling_module}'."
        )


class InvalidModel(Exception):
    model: Type

    def __init__(
        self,
        model,
    ):
        super().__init__(
            f"'model' accepts either a DynapcnnNetwork or a DynapcnnNetworkGraph but {model} was given."
        )


class InvalidTorchModel(Exception):
    network_type: str

    def __init__(self, network_type):
        super().__init__(f"A {network_type} needs to be of type nn.Module.")


class InvalidGraphStructure(Exception):
    pass


class InvalidModelWithDVSSetup(Exception):
    def __init__(self):
        super().__init__(f"The network provided has a DVSLayer instance but argument 'dvs_input' is set to False.")

# Edge exceptions.


class InvalidEdge(Exception):
    edge: Tuple[int, int]
    source: Type
    target: Type

    def __init__(self, edge, source, target):
        super().__init__(f"Invalid edge {edge}: {source} can not target {target}.")


class UnknownNode(Exception):
    node: int

    def __init__(self, node):
        super().__init__(
            f"Node {node} can not be found within any DynapcnnLayer mapper."
        )


class MaxDestinationsReached(Exception):
    dynapcnnlayer_index: int

    def __init__(self, dynapcnnlayer_index):
        super().__init__(
            f"DynapcnnLayer with index {dynapcnnlayer_index} has more than 2 destinations."
        )


class InvalidLayerLoop(Exception):
    dynapcnnlayerA_index: int
    dynapcnnlayerB_index: int

    def __init__(self, dynapcnnlayerA_index, dynapcnnlayerB_index):
        super().__init__(
            f"DynapcnnLayer {dynapcnnlayerA_index} can not connect to {dynapcnnlayerB_index} since reverse edge already exists."
        )


class InvalidLayerDestination(Exception):
    dynapcnnlayerA: Type
    dynapcnnlayerB: Type

    def __init__(self, dynapcnnlayerA, dynapcnnlayerB):
        super().__init__(
            f"DynapcnnLayer {dynapcnnlayerA} in one core can not connect to {dynapcnnlayerB} in another core."
        )
