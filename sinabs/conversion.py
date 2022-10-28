import torch.nn as nn
from typing import Callable
from copy import deepcopy


def replace_module(model: nn.Module, source_class: type, mapper_fn: Callable):
    """
    A utility function that returns a copy of the model, where specific layers are replaced with
    another type depending on the mapper function.

    Parameters:
        model: A PyTorch model.
        source_class: the layer class to replace. Each find will be passed to mapper_fn
        mapper_fn: A callable that takes as argument the layer to replace and returns the new object.

    Returns:
        A model copy with replaced modules according to mapper_fn.
    """
    new_model = deepcopy(model)
    replace_module_(new_model, source_class, mapper_fn)
    return new_model


def replace_module_(model: nn.Sequential, source_class: type, mapper_fn: Callable):
    """
    In-place version of replace_module that will step through modules that have children and
    apply the mapper_fn.

    Parameters:
        model: A PyTorch model.
        source_class: the layer class to replace. Each find will be passed to mapper_fn
        mapper_fn: A callable that takes as argument the layer to replace and returns the new object.

    Returns:
        None. Model is modified in-place.
    """
    for name, module in model.named_children():
        if list(module.named_children()):
            replace_module_(module, source_class, mapper_fn)

        if type(module) == source_class:
            setattr(model, name, mapper_fn(module))
