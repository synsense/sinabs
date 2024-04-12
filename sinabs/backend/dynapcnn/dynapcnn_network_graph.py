# functionality : ...
# author        : Willian Soares Girao
# contact       : williansoaresgirao@gmail.com

import time
from subprocess import CalledProcessError
from typing import List, Optional, Sequence, Tuple, Union

import samna
import sinabs.layers
import torch
import torch.nn as nn

import sinabs

from .chip_factory import ChipFactory
from .dvs_layer import DVSLayer
from .dynapcnn_layer import DynapcnnLayer
from .io import disable_timestamps, enable_timestamps, open_device, reset_timestamps
from .utils import (
    DEFAULT_IGNORED_LAYER_TYPES,
    build_from_list,
    build_from_graph,
    convert_model_to_layer_list,
    infer_input_shape,
    parse_device_id,
)

from .graph_tracer import GraphTracer

class DynapcnnNetworkGraph(nn.Module):
    """Given a sinabs spiking network, prepare a dynapcnn-compatible network. This can be used to
    test the network will be equivalent once on DYNAPCNN. This class also provides utilities to
    make the dynapcnn configuration and upload it to DYNAPCNN.
    """

    def __init__(
        self,
        snn: Union[nn.Sequential, sinabs.Network],
        input_shape: Optional[Tuple[int, int, int]] = None,
        dvs_input: bool = False,
        discretize: bool = True
    ):
        """
        DynapcnnNetworkGraph: a class turning sinabs networks into dynapcnn
        compatible networks, and making dynapcnn configurations.

        Parameters
        ----------
            snn: sinabs.Network
                SNN that determines the structure of the `DynapcnnNetwork`
            input_shape: None or tuple of ints
                Shape of the input, convention: (features, height, width)
                If None, `snn` needs an InputLayer
            dvs_input: bool
                Does dynapcnn receive input from its DVS camera?
            discretize: bool
                If True, discretize the parameters and thresholds.
                This is needed for uploading weights to dynapcnn. Set to False only for
                testing purposes.
        """
        super().__init__()

        dvs_input = False                                           # @TODO for now the graph part is not taking into consideration this.

        self.graph_tracer = GraphTracer(                            # computational graph from original PyTorch module.
            snn.analog_model, 
            torch.randn((1, *input_shape))                          # torch.jit needs the batch dimension.
            )

        self.input_shape = input_shape                              # convert models  to sequential.
        self.layers = convert_model_to_layer_list(
            model=snn.spiking_model, ignore=DEFAULT_IGNORED_LAYER_TYPES
        )

        self.dvs_input = dvs_input                                  # check if dvs input is expected.

        input_shape = infer_input_shape(self.layers, input_shape=input_shape)
        assert len(input_shape) == 3, "infer_input_shape did not return 3-tuple"

        self.sinabs_edges = self.get_sinabs_edges(snn)              # get sinabs graph.

        self.dynapcnn_layers, \
            self.nodes_to_dcnnl_map, \
                self.dcnnl_to_dcnnl_map = build_from_graph(         # build model from graph edges.
            discretize=discretize,
            layers=self.layers, 
            in_shape=input_shape,
            edges=self.sinabs_edges)

    def __str__(self):
        pretty_print = ''
        for idx, layer_dest in self.dynapcnn_layers.items():
            layer = layer_dest['layer']
            dest = layer_dest['destinations']
            pretty_print += f'\nlayer index: {idx}\nlayer modules: {layer}\nlayer destinations: {dest}\n'
        return pretty_print
        
    @staticmethod
    def build_from_graph_():                                        # @TODO used for debug only (remove when class is complete).
        return build_from_graph
    
    def to(
        self,
        device="cpu",
        chip_layers_ordering="auto",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier=None,
        slow_clk_frequency: int = None,
    ):
        """ ."""
        self.device = device

        if isinstance(device, torch.device):
            return super().to(device)
        
        elif isinstance(device, str):
            device_name, _ = parse_device_id(device)

            if device_name in ChipFactory.supported_devices:        # pragma: no cover
                
                config = self.make_config(                          # generate config.
                    chip_layers_ordering=chip_layers_ordering,
                    device=device,
                    monitor_layers=monitor_layers,
                    config_modifier=config_modifier,
                )

    def make_config(
        self,
        chip_layers_ordering: Union[Sequence[int], str] = "auto",
        device="dynapcnndevkit:0",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier=None,
    ):
        """Prepare and output the `samna` DYNAPCNN configuration for this network.

        Parameters
        ----------

        chip_layers_ordering: sequence of integers or `auto`
            The order in which the dynapcnn layers will be used. If `auto`,
            an automated procedure will be used to find a valid ordering.
            A list of layers on the device where you want each of the model's DynapcnnLayers to be placed.
            Note: This list should be the same length as the number of dynapcnn layers in your model.

        device: String
            dynapcnndevkit, speck2b or speck2devkit

        monitor_layers: None/List/Str
            A list of all layers in the module that you want to monitor. Indexing starts with the first non-dvs layer.
            If you want to monitor the dvs layer for eg.
            ::

                monitor_layers = ["dvs"]  # If you want to monitor the output of the pre-processing layer
                monitor_layers = ["dvs", 8] # If you want to monitor preprocessing and layer 8
                monitor_layers = "all" # If you want to monitor all the layers

            If this value is left as None, by default the last layer of the model is monitored.

        config_modifier:
            A user configuration modifier method.
            This function can be used to make any custom changes you want to make to the configuration object.

        Returns
        -------
        Configuration object
            Object defining the configuration for the device

        Raises
        ------
            ImportError
                If samna is not available.
            ValueError
                If the generated configuration is not valid for the specified device.
        """
        config, is_compatible = self._make_config(
            chip_layers_ordering=chip_layers_ordering,
            device=device,
            monitor_layers=monitor_layers,
            config_modifier=config_modifier,
        )

        if is_compatible:                                           # validate config.
            print("Network is valid")
            return config
        else:
            raise ValueError(f"Generated config is not valid for {device}")
        
    def _make_config(
        self,
        chip_layers_ordering: Union[Sequence[int], str] = "auto",
        device="dynapcnndevkit:0",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier=None,
    ) -> Tuple["SamnaConfiguration", bool]:
        """ Prepare and output the `samna` configuration for this network. """

        config_builder = ChipFactory(device).get_config_builder()

        has_dvs_layer = isinstance(self.dynapcnn_layers[0]['layer'], DVSLayer)

        if chip_layers_ordering == "auto":                           # figure out mapping of each DynapcnnLayer into one core.
            chip_layers_ordering = config_builder.get_valid_mapping(self)

        else:                                                        # mapping from each DynapcnnLayer into cores has been provided.
            if has_dvs_layer:                                        # @TODO maybe this has to be modified given the new representation of layers in a dictionary (instead of a list).
                chip_layers_ordering = chip_layers_ordering[: len(self.sequence) - 1]

            chip_layers_ordering = chip_layers_ordering[: len(self.sequence)]
    
    def get_sinabs_edges(self, sinabs_model):
        """ Converts the computational graph extracted from 'sinabs_model.analog_model' into its equivalent
        representation for the 'sinabs_model.spiking_model'.
        
        Parameters
        ----------
            sinabs_model: ...

        Returns
            sinabs_edges: ...
        ----------
        """
        # parse original graph to ammend edges containing nodes dropped in 'convert_model_to_layer_list()'.
        sinabs_edges = self.graph_tracer.remove_ignored_nodes(DEFAULT_IGNORED_LAYER_TYPES)

        if DynapcnnNetworkGraph.was_spiking_output_added(sinabs_model):
            # spiking output layer has been added: create new edge.
            last_edge = sinabs_edges[-1]
            new_edge = (last_edge[1], last_edge[1]+1)
            sinabs_edges.append(new_edge)
        else:
            pass

        return sinabs_edges

    @staticmethod
    def was_spiking_output_added(sinabs_model):
        """ Compares the models outputed by 'sinabs.from_torch.from_model()' to check if
        a spiking output was added to the spiking version of the analog model.
        """
        analog_modules = []
        spiking_modules = []

        for mod in sinabs_model.analog_model:
            analog_modules.append(mod)

        for mod in sinabs_model.spiking_model:
            spiking_modules.append(mod)

        if len(analog_modules) != len(spiking_modules):
            if isinstance(spiking_modules[-1], sinabs.layers.iaf.IAFSqueeze):
                return True
            else:
                # throw error
                return False
        else:
            return False