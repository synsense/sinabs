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
        
        1. @TODO | 'self.sinabs_edges' has to be validated to be abiding to the device contraints before being passed to 'build_from_graph()'.
        """
        super().__init__()

        dvs_input = False                                           # @TODO for now the graph part is not taking into consideration this.

        # Computational graph from original PyTorch module.
        self.graph_tracer = GraphTracer(
            snn.analog_model, 
            torch.randn((1, *input_shape))                          # torch.jit needs the batch dimension.
            )

        # This attribute stores the location/core-id of each of the DynapcnnLayers upon placement on chip.
        self.chip_layers_ordering = []

        self.input_shape = input_shape                              # convert models  to sequential.
        self.layers = convert_model_to_layer_list(
            model=snn.spiking_model, ignore=DEFAULT_IGNORED_LAYER_TYPES
        )

        # Check if dvs input is expected.
        if dvs_input:
            self.dvs_input = True
        else:
            self.dvs_input = False

        input_shape = infer_input_shape(self.layers, input_shape=input_shape)
        assert len(input_shape) == 3, "infer_input_shape did not return 3-tuple"

        # Build model from layers.
        self.sequence = build_from_list(
            self.layers,
            in_shape=input_shape,
            discretize=discretize,
            dvs_input=self.dvs_input,
        )

        # Get sinabs graph.
        self.sinabs_edges = self.get_sinabs_edges(snn)

        _ = build_from_graph(
            layers=self.layers, 
            in_shape=input_shape,
            edges=self.sinabs_edges)

    def get_sinabs_edges(self, sinabs_model):
        """ Converts the computational graph extracted from 'sinabs_model.analog_model' into its equivalent
        representation for the 'sinabs_model.spiking_model'.
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