# functionality : ...
# author        : Willian Soares Girao
# contact       : williansoaresgirao@gmail.com

import torch.nn as nn
from sinabs.layers import Merge
from typing import List, Tuple, Dict, Union
import copy
import sinabs
import sinabs.layers as sl

class DynapcnnNetworkModule(nn.Module):
    """
        Uses the set of `DynapcnnLayer` instances that have been configured to the chip and how they address each other
    to define what the `forward` method of the model should do.

    Parameters
    ----------
        dcnnl_edges (list): tuples representing the output->input mapping between `DynapcnnLayer` instances
            that have been used as configuration for each core `CNNLayerConifg`.
        dynapcnn_layers (dict): the `DynapcnnLayer` instances along with their supporting metadata (e.g. assigned core,
            destination layers, etc.).
    """

    def __init__(self, dcnnl_edges: List[Tuple[int, int]], dynapcnn_layers: Dict) -> nn.Module:
        super().__init__()

        self.dcnnl_edges = dcnnl_edges

        self.forward_map, self.merge_points = self._build_module_forward_from_graph_v2(dcnnl_edges, dynapcnn_layers)
    
    def _spot_merging_points(self, dcnnl_edges: list) -> dict:
        """ . """

        nodes_with_merge_input = {}

        for edge in dcnnl_edges:
            trg_node = edge[1]
            fan_in = 0
            src_nodes = []

            for edge_inner in dcnnl_edges:
                if edge_inner[1] == trg_node:
                    fan_in += 1
                    src_nodes.append(edge_inner[0])

            if fan_in == 2 and trg_node not in nodes_with_merge_input:
                nodes_with_merge_input[trg_node] = {'sources': tuple(src_nodes), 'merge': sl.Merge()}
            
            if fan_in > 2:
                raise ValueError(f'Node {trg_node} is the has fan-in of {fan_in}: only fan-in of 2 is currently handled.')
            
        return nodes_with_merge_input

    
    def _build_module_forward_from_graph_v2(self, dcnnl_edges: list, dynapcnn_layers: dict) -> Union[dict, dict]:
        """ ."""

        # mapper to flag nodes that need input from a `Merge` layer.
        merge_points = self._spot_merging_points(dcnnl_edges)

        # this dict. will be used to call the `forward` methods of each `DynapcnnLayer`.
        forward_map = {}

        for edge in dcnnl_edges:
            src_dcnnl = edge[0]     # source layer
            trg_dcnnl = edge[1]     # target layer

            if src_dcnnl not in forward_map:
                forward_map[src_dcnnl] = copy.deepcopy(dynapcnn_layers[src_dcnnl]['layer'])
            
            if trg_dcnnl not in forward_map:
                forward_map[trg_dcnnl] = copy.deepcopy(dynapcnn_layers[trg_dcnnl]['layer'])

        return forward_map, merge_points
    
    def forward(self, x):
        """ ."""

        layers_outputs = {}

        #   TODO - currently `node 0` (this 1st node in the 1st edge of `self.dcnnl_edges`) is always taken to be the
        # input node of the network. This won't work in cases where there are more the one input nodes to the network
        # so this functionality needs some refactoring.
        self.forward_map[self.dcnnl_edges[0][0]](x)

        # forward the input `x` through the input `DynapcnnLayer` in the `DynapcnnNetwork`s graph (1st node in the 1st edge in `self.dcnnl_edges`).
        layers_outputs[self.dcnnl_edges[0][0]] = self.forward_map[self.dcnnl_edges[0][0]](x)

        # propagate outputs in `layers_outputs` through the rest of the nodes of `self.dcnnl_edges`.
        for edge in self.dcnnl_edges:
            
            # target DynapcnnLayer (will consume tensors from `layers_outputs`).
            trg_dcnnl = edge[1]

            if trg_dcnnl in self.merge_points and trg_dcnnl not in layers_outputs:
                # by this points the arguments of the `Merge` associated with `trg_dcnnl` should have been computed.
                arg1, arg2 = self.merge_points[trg_dcnnl]['sources']

                #   find which returned tensor from the `forward` call of DynapcnnLayers `arg1` and `arg2` are to be fed
                # to the target DynapcnnLayer `trg_dcnnl`.
                return_index_arg1 = self.forward_map[arg1].get_destination_dcnnl_index(trg_dcnnl)
                return_index_arg2 = self.forward_map[arg2].get_destination_dcnnl_index(trg_dcnnl)

                # retrieve input tensors to `Merge`.
                _arg1 = layers_outputs[arg1][return_index_arg1]
                _arg2 = layers_outputs[arg2][return_index_arg2]

                # merge tensors.
                merge_output = self.merge_points[trg_dcnnl]['merge'](_arg1, _arg2)

                # call the forward.
                layers_outputs[trg_dcnnl] = self.forward_map[trg_dcnnl](merge_output)

            elif trg_dcnnl not in layers_outputs:
                # input source for `trg_dcnnl`.
                src_dcnnl = edge[0]

                #   find which returned tensor from the `forward` call of the source DynapcnnLayer `src_dcnnl` is to be fed
                # to the target DynapcnnLayer `trg_dcnnl`.
                return_index = self.forward_map[src_dcnnl].get_destination_dcnnl_index(trg_dcnnl)

                # call the forward.
                layers_outputs[trg_dcnnl] = self.forward_map[trg_dcnnl](layers_outputs[src_dcnnl][return_index])

            else:

                pass
        
        # TODO - this assumes the network has a single output node.
        # last computed is the output layer.
        return layers_outputs[trg_dcnnl][0]
    
    def parameters(self) -> list:
        """ Gathers all the parameters of the network in a list. This is done by accessing the convolutional layer in each `DynapcnnLayer`, calling 
        its `.parameters` method and saving it to a list.

        Note: the method assumes no biases are used.

        Returns
        ----------
            parameters (list): a list of parameters of all convolutional layers in the `DynapcnnNetwok`.
        """
        parameters = []

        for layer in self.forward_map.values():
            if isinstance(layer, sinabs.backend.dynapcnn.dynapcnn_layer_new.DynapcnnLayer):
                parameters.extend(layer.conv_layer.parameters())

        return parameters
    
    def init_weights(self, init_fn: nn.init = nn.init.xavier_normal_) -> None:
        """ Call the weight initialization method `init_fn` on each `DynapcnnLayer.conv_layer.weight.data` in the `DynapcnnNetwork` instance."""
        for layer in self.forward_map.values():
            if isinstance(layer, sinabs.backend.dynapcnn.dynapcnn_layer_new.DynapcnnLayer):
                init_fn(layer.conv_layer.weight.data)

    def to(self, device) -> None:
        """ ."""
        for layer in self.forward_map.values():
            if isinstance(layer, sinabs.backend.dynapcnn.dynapcnn_layer_new.DynapcnnLayer):
                layer.conv_layer.to(device)
                layer.spk_layer.to(device)
                
                # if there's more than one pooling each of them becomes a node that is catched by the `else` statement.
                if len(layer.pool_layer) == 1:
                    layer.pool_layer[0].to(device)
            else:
                # this nodes are created from `DynapcnnLayer`s that have multiple poolings (each pooling becomes a new node).
                layer.to(device)

    def detach_neuron_states(self) -> None:
        """ Detach the neuron states and activations from current computation graph (necessary). """

        for module in self.forward_map.values():
            if isinstance(module, sinabs.backend.dynapcnn.dynapcnn_layer_new.DynapcnnLayer):
                if isinstance(module.spk_layer, sl.StatefulLayer):
                    for name, buffer in module.spk_layer.named_buffers():
                        buffer.detach_()