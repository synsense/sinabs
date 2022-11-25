from copy import deepcopy
from typing import Callable
from warnings import warn

import torch.nn as nn


def replace_module(model: nn.Module, source_class: type, mapper_fn: Callable):
    """A utility function that returns a copy of the model, where specific layers are replaced with
    another type depending on the mapper function.

    Parameters:
        model: A PyTorch model.
        source_class: the layer class to replace. Each find will be passed to mapper_fn
        mapper_fn: A callable that takes as argument the layer to replace and returns the new object.

    Returns:
        A model copy with replaced modules according to mapper_fn.
    """

    # Handle case where `model` is of type `source_class`
    if type(model) == source_class:
        return mapper_fn(model)

    new_model = deepcopy(model)
    replace_module_(new_model, source_class, mapper_fn)
    return new_model


def replace_module_(model: nn.Sequential, source_class: type, mapper_fn: Callable):
    """In-place version of replace_module that will step through modules that have children and
    apply the mapper_fn.

    Parameters:
        model: A PyTorch model.
        source_class: the layer class to replace. Each find will be passed to mapper_fn
        mapper_fn: A callable that takes as argument the layer to replace and returns the new object.

    Returns:
        Modified model.
    """

    # If `model` is of type `source_class`, it cannot be converted in-place.
    if type(model) == source_class:
        warn(
            f"Provided model is of type `{source_class}` and cannot be converted"
            " in-place if not part of another Module. Apply mapper function"
            " directly or use `replace_module` to generate new object of desired type."
        )

    for name, module in model.named_children():
        if list(module.named_children()):
            replace_module_(module, source_class, mapper_fn)

        if type(module) == source_class:
            setattr(model, name, mapper_fn(module))
