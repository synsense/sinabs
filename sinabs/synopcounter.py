import torch
from sinabs.layers import NeuromorphicReLU


class SynOpCounter(object):
    """
    Counter for the synaptic operations emitted by all Neuromorphic ReLUs in a model.

    Usage:
        counter = SynOpCounter(MyTorchModel.modules(), sum_activations=True)
        output = MyTorchModule(input)  # forward pass
        synop_count = counter()

    :param modules: list of modules, e.g. MyTorchModel.modules()
    :param sum_activations: If True (default), returns a single number of synops, otherwise a list of layer synops.
    """

    def __init__(self, modules, sum_activations=True):
        self.modules = []
        for module in modules:
            if isinstance(module, NeuromorphicReLU) and module.fanout > 0:
                self.modules.append(module)

        if len(self.modules) == 0:
            raise ValueError("No NeuromorphicReLU found in module list.")

        self.sum_activations = sum_activations
        # self.modules[1:] = []

    def __call__(self):
        synops = []
        for module in self.modules:
            synops.append(module.activity)

        if self.sum_activations:
            synops = torch.stack(synops).sum()
        return synops
