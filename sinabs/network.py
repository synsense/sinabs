import warnings

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict
from .utils import get_network_activations, get_activations
from .layers import StatefulLayer
from .synopcounter import SNNSynOpCounter

ArrayLike = Union[np.ndarray, List, Tuple]


class Network(torch.nn.Module):
    """
    Class of a spiking neural network

    Attributes:
        spiking_model: torch.nn.Module, a spiking neural network model
        analog_model: torch.nn.Module, an artifical neural network model
        input_shape: Tuple, size of input
        synops: If True (default: False), register hooks for counting synaptic \
        operations during forward passes, instantiating `sinabs.SNNSynOpCounter`.
    """

    def __init__(
        self,
        analog_model=None,
        spiking_model=None,
        input_shape: Optional[ArrayLike] = None,
        synops: bool = False,
        batch_size: int = 1,
        num_timesteps: int = 1,
    ):
        super().__init__()
        self.spiking_model: nn.Module = spiking_model
        self.analog_model: nn.Module = analog_model
        self.input_shape = input_shape

        self.synops = synops
        if synops:
            self.synops_counter = SNNSynOpCounter(self.spiking_model)

        if input_shape is not None and spiking_model is not None:
            self._compute_shapes(
                input_shape, batch_size=batch_size, num_timesteps=num_timesteps
            )

    @property
    def layers(self):
        return list(self.spiking_model.named_children())

    def _compute_shapes(self, input_shape, batch_size=1, num_timesteps=1):
        def hook(module, inp, out):
            module.out_shape = out.shape[1:]

        hook_list = []
        for layer in self.spiking_model.modules():
            this_hook = layer.register_forward_hook(hook)
            hook_list.append(this_hook)

        device = next(self.parameters()).device

        # Infer shape
        if batch_size is None:
            batch_size = 1
        if num_timesteps is None:
            num_timesteps = 1
        shape = [batch_size * num_timesteps] + list(input_shape)
        dummy_input = torch.zeros(shape, requires_grad=False).to(device)
        # do a forward pass
        self(dummy_input)

        [this_hook.remove() for this_hook in hook_list]

    def forward(self, tsrInput) -> torch.Tensor:
        """
        Forward pass for this model
        """
        return self.spiking_model(tsrInput)

    def compare_activations(
        self,
        data,
        name_list: Optional[ArrayLike] = None,
        compute_rate: bool = False,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Compare activations of the analog model and the SNN for a given data sample

        Args:
            data (np.ndarray):      Data to process
            name_list (List[str]):  list of all layer names (str) whose activations need to be compared
            compute_rate (bool):    True if you want to compute firing rate. By default spike count is returned
            verbose (bool):         print debugging logs to the terminal
        Returns:
            tuple: A tuple of lists (ann_activity, snn_activity, name_list)
                - ann_activity: output activity of the ann layers
                - snn_activity: output activity of the snn layers
                - name_list: spiking layers' name list for plotting comparison
        """
        if name_list is None:
            name_list = ["Input"]
            for layer_name, lyr in self.spiking_model.named_modules():
                if isinstance(lyr, StatefulLayer):
                    name_list.append(layer_name)

        if verbose:
            print("Comparing activations for {0}".format(name_list))

        # Calculate activations for the torch analog model
        if compute_rate:
            tsrAnalogData = data.mean(0).unsqueeze(0)
        else:
            tsrAnalogData = data.sum(0).unsqueeze(0)
        with torch.no_grad():
            analog_activations = get_activations(
                self.analog_model, tsrAnalogData, name_list=name_list
            )

        # Calculate activations for spiking model
        spike_rates = get_network_activations(
            self.spiking_model, data, name_list=name_list, bRate=compute_rate
        )
        return analog_activations, spike_rates, name_list

    def plot_comparison(
        self, data, name_list: Optional[ArrayLike] = None, compute_rate=False
    ):
        """
        Plots a scatter plot of all the activations

        Args:
            data: Data to be processed
            name_list: ArrayLike with names of all the layers of interest to be compared
            compute_rate: Compare firing rates instead of spike count
        Returns:
            tuple: A tuple of lists (ann_activity, snn_activity)
                - ann_activity: output activity of the ann layers
                - snn_activity: output activity of the snn layers
        """
        import pylab

        analog_activations, spike_rates, name_list = self.compare_activations(
            data, name_list=name_list, compute_rate=compute_rate
        )
        for nLyrIdx in range(len(name_list)):
            pylab.scatter(
                spike_rates[nLyrIdx],
                analog_activations[nLyrIdx],
                label=name_list[nLyrIdx],
            )
        if compute_rate:
            pylab.xlabel("Spike rates (Hz)")
        else:
            pylab.xlabel("# Spike count")
        pylab.ylabel("Analog activations")
        pylab.legend()
        return analog_activations, spike_rates

    def reset_states(
        self,
        randomize: bool = False,
        value_ranges: Optional[List[Dict[str, Tuple[float, float]]]] = None,
    ):
        """
        Reset all neuron states in the submodules.

        Parameters
        ----------
        randomize: Bool
            If true, reset the states between a range provided. Else, the states are reset to zero.
        value_ranges: Optional[List[Dict[str, Tuple[float, float]]]]
            A list of value_range dictionaries with the same length as the total stateful layers in the module.
            Each dictionary is a key value pair: buffer_name -> (min, max) for each state that needs to be reset.
            The states are reset with a uniform distribution between the min and max values specified.
            Any state with an undefined key in this dictionary will be reset between 0 and 1
            This parameter is only used if randomize is set to true.
        """

        if value_ranges:
            num_stateful_layers = len(
                [None for mod in self.modules() if isinstance(mod, StatefulLayer)]
            )
            if len(value_ranges) != num_stateful_layers:
                raise TypeError(
                    "The number of entries in value_ranges does not match the number of stateful sub modules"
                )
        i = 0
        for lyr in self.modules():
            if isinstance(lyr, StatefulLayer):
                if value_ranges is None:
                    vr = None
                else:
                    vr = value_ranges[i]
                    i += 1
                lyr.reset_states(randomize=randomize, value_ranges=vr)

    def get_synops(self, num_evs_in=None) -> pd.DataFrame:
        """
        Please see docs for `sinabs.SNNSynOpCounter.get_synops()`.
        """
        if num_evs_in is not None:
            warnings.warn("num_evs_in is deprecated and has no effect")

        return self.synops_counter.get_synops()


def get_parent_module_by_name(
    root: torch.nn.Module, name: str
) -> Tuple[torch.nn.Module, str]:
    """
    Find a nested Module of a given name inside a Module, and return its parent
    Module.

    Args:
        root: The Module inside which to look for the nested Module
        name: Name of the Module that is being searched for within root. Must
              contain all parent modules, separated by a `.` , e.g.
              "root.nested_module1.nested_module2.desired_module"
    Returns:
        torch.nn.Module: The Module that contains the Module with the given name. In the example
        above this would be `nested_module2`.
        str: The name of the child, without parent modules, e.g. "desired_module"
    """
    if "." not in name:
        if not hasattr(root, name):
            raise KeyError(f"The requested module `{name}` could not be found.")

        return root, name
    else:
        child_name, *rest = name.split(".")
        child = getattr(root, child_name)
        return get_parent_module_by_name(child, ".".join(rest))


def infer_module_device(module: torch.nn.Module) -> Union[torch.device, None]:
    """
    Infere on which device a module is operating by first looking at its parameters
    and then, if no parameters are found, at its buffers.

    Args:
        module: The module whose device is to be inferred.

    Returns:
        torch.device: The device of 'module', or `None` if no device has been found.
    """

    try:
        return next(module.parameters()).device
    except StopIteration:
        # No parameters, try buffers
        try:
            return next(module.buffers()).device
        except StopIteration:
            # No buffers, don't infer device
            return None
