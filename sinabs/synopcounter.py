import torch
from sinabs.layers import NeuromorphicReLU
from numpy import product
from warnings import warn
import pandas as pd


def synops_hook(layer, inp, out):
    assert len(inp) == 1, "Multiple inputs not supported for synops hook"
    inp = inp[0]
    layer.tot_in = inp.sum().item()
    layer.tot_out = out.sum().item()
    layer.synops = layer.tot_in * layer.fanout
    layer.tw = inp.shape[0]


class SNNSynOpCounter:
    def __init__(self, model):
        self.model = model
        self.handles = []

        for layer in model.modules():
            self.register_synops_hook(layer)

    def register_synops_hook(self, layer):
        if isinstance(layer, torch.nn.Conv2d):
            layer.fanout = (layer.out_channels *
                            product(layer.kernel_size) /
                            product(layer.stride))
        elif isinstance(layer, torch.nn.Linear):
            layer.fanout = layer.out_features
        else:
            return None

        handle = layer.register_forward_hook(synops_hook)
        self.handles.append(handle)

    def get_synops(self) -> pd.DataFrame:
        SynOps_dataframe = pd.DataFrame()
        for i, lyr in enumerate(self.model.modules()):
            if hasattr(lyr, 'synops'):
                SynOps_dataframe = SynOps_dataframe.append(
                    pd.Series(
                        {
                            "Layer": i,
                            "In": lyr.tot_in,
                            "Fanout_Prev": lyr.fanout,
                            "SynOps": lyr.synops,
                            "Time_window": lyr.tw,
                            "SynOps/s": lyr.synops / lyr.tw * 1000,
                        }
                    ),
                    ignore_index=True,
                )
        SynOps_dataframe.set_index("Layer", inplace=True)
        return SynOps_dataframe

    def get_total_power_use(self, j_per_synop=1e-11):
        synops_table = self.get_synops()
        tot_synops_per_s = synops_table["SynOps/s"].sum()
        power_in_mW = tot_synops_per_s * j_per_synop * 1000
        return power_in_mW

    def __del__(self):
        for handle in self.handles:
            handle.remove()


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
