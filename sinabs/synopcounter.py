import warnings

import torch
from sinabs.layers import NeuromorphicReLU
from numpy import product
import pandas as pd


def synops_hook(layer, inp, out):
    assert len(inp) == 1, "Multiple inputs not supported for synops hook"
    inp = inp[0]
    layer.tot_in = inp.sum()
    layer.tot_out = out.sum()
    layer.synops = layer.tot_in * layer.fanout
    layer.tw = inp.shape[0]


class SNNSynOpCounter:
    """
    Counter for the synaptic operations emitted by all SpikingLayers in a
    spiking model.
    Note that this is automatically instantiated by `from_torch` and by
    `Network` if they are passed `synops=True`.

    Usage:
        counter = SNNSynOpCounter(my_spiking_model)

        output = my_spiking_model(input)  # forward pass

        synops_table = counter.get_synops()

    Arguments:
        model: Spiking model.
        dt: the number of milliseconds corresponding to a time step in the \
        simulation (default 1.0).
    """

    def __init__(self, model, dt=1.0):
        self.model = model
        self.handles = []
        self.dt = dt

        for layer in model.modules():
            self._register_synops_hook(layer)

    def _register_synops_hook(self, layer):
        if isinstance(layer, torch.nn.Conv2d):
            layer.fanout = (
                layer.out_channels * product(layer.kernel_size) / product(layer.stride)
            )
        elif isinstance(layer, torch.nn.Linear):
            layer.fanout = layer.out_features

        else:
            return None

        handle = layer.register_forward_hook(synops_hook)
        self.handles.append(handle)

    def get_synops(self) -> pd.DataFrame:
        """
        Method to compute a table of synaptic operations for the latest forward pass.

        NOTE: this may not be accurate in presence of average pooling.

        Returns:
            SynOps_dataframe: A Pandas DataFrame containing layer IDs and \
            respectively, for the latest forward pass performed, their:
                number of input spikes,
                fanout,
                synaptic operations,
                number of timesteps,
                total duration of simulation,
                number of synaptic operations per second.
        """
        d = {}
        scale_facts = []
        for i, lyr in enumerate(self.model.modules()):
            if isinstance(lyr, torch.nn.AvgPool2d):
                if lyr.kernel_size != lyr.stride:
                    warnings.warn(
                        f"In order for the Synops counter to work accurately the pooling "
                        f"layers kernel size should match their strides. At the moment at layer {i}, "
                        f"the kernel_size = {lyr.kernel_size}, the stride = {lyr.stride}."
                    )
                ks = lyr.kernel_size
                scale_factor = ks**2 if isinstance(ks, int) else ks[0] * ks[1]
                scale_facts.append(scale_factor)
            if hasattr(lyr, "synops"):
                scale_factor = 1
                while len(scale_facts) != 0:
                    scale_factor *= scale_facts.pop()
                d[i] = {
                    "Layer": i,
                    "In": lyr.tot_in * scale_factor,
                    "Fanout_Prev": lyr.fanout,
                    "SynOps": lyr.synops * scale_factor,
                    "N. timesteps": lyr.tw,
                    "Time window (ms)": lyr.tw * self.dt,
                    "SynOps/s": (lyr.synops * scale_factor) / lyr.tw / self.dt * 1000,
                }

        SynOps_dataframe = pd.DataFrame.from_dict(d, "index")
        SynOps_dataframe.set_index("Layer", inplace=True)
        return SynOps_dataframe

    def get_total_synops(self, per_second=False) -> float:
        """
        Faster method for computing total synaptic operations without using Pandas.

        NOTE: this may not be accurate in presence of average pooling.

        Arguments:
            per_second (bool, default False): if True, gives synops per second \
        instead of total synops in the last forward pass.

        Returns:
            synops: the total synops in the network, based on the last forward pass.
        """
        synops = 0.0
        for i, lyr in enumerate(self.model.modules()):
            if hasattr(lyr, "synops"):
                if per_second:
                    layer_synops = lyr.synops / lyr.tw / self.dt * 1000
                else:
                    layer_synops = lyr.synops

                synops = synops + layer_synops
        return synops

    def get_total_power_use(self, j_per_synop=1e-11):
        """
        Method to quickly get the total power use of the network, estimated
        over the latest forward pass.

        Arguments:
            j_per_synop: Energy use per synaptic operation, in joules.\
            Default 1e-11 J.

        Returns: estimated power in mW.
        """
        tot_synops_per_s = self.get_total_synops(per_second=True)
        power_in_mW = tot_synops_per_s * j_per_synop * 1000
        return power_in_mW

    def __del__(self):
        for handle in self.handles:
            handle.remove()


class SynOpCounter:
    """
    Counter for the synaptic operations emitted by all Neuromorphic ReLUs in an
    analog CNN model.

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
