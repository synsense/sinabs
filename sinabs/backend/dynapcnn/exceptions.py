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
    """Is raised when input to dynapcnn is not configured correctly"""

    pass
