class MissingLayer(Exception):
    index: int

    def __init__(self, index: int):
        super().__init__(f"A spiking layer is expected at index {index}")


class UnexpectedLayer(Exception):
    layer_type_found: type
    layer_type_expected: type

    def __init__(self, expected, found):
        super().__init__(f"Expected {expected} but found {found}")


class InputConfigurationError(Exception):
    """Is raised when input to dynapcnn is not configured correctly."""

    pass

class WrongModuleCount(Exception):
    dynapcnnlayer_indx: type
    modules_count: type

    def __init__(self, dynapcnnlayer_indx, modules_count):
        super().__init__(f"A DynapcnnLayer {dynapcnnlayer_indx} should have 2 or 3 modules but found {modules_count}.")

class WrongPoolingModule(Exception):
    pooling_module: type

    def __init__(self, pooling_module,):
        super().__init__(f"The function 'utils.build_SumPool2d(mod)' expects 'mod = nn.AvgPool2d' but got 'mod = {pooling_module}'.")
