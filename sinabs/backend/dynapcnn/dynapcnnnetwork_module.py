# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

import copy
from typing import Dict, List, Tuple, Union

import torch.nn as nn

import sinabs.layers as sl

from .dynapcnn_layer import DynapcnnLayer


class DynapcnnNetworkModule:
    """
    Uses the set of `DynapcnnLayer`\`DynapcnnLayerHandler` instances and how they address each other to define what the `forward` method of the model should do.

    Parameters
    ----------
    - dcnnl_edges (list): tuples representing the output->input mapping between `DynapcnnLayer` instances
        that have been used as configuration for each core `CNNLayerConifg`.
    - dynapcnn_layers (dict): a mapper containing `DynapcnnLayer` instances.
    - dynapcnnlayers_handlers (dict): a mapper containing `DynapcnnLayerHandler` instances (hold network-level
        data that was used to create the respective `DynapcnnLayer` instances in `dynapcnn_layers`).
    """

    def __init__(
        self,
        dcnnl_edges: List[Tuple[int, int]],
        dynapcnn_layers: Dict[int, dict],
        dynapcnnlayers_handlers: Dict[int, dict],
    ):

        self.dcnnl_edges = dcnnl_edges

        # create mappers to handle `DynapcnnLayer` instances' forward calling.
        self.forward_map, self.merge_points = self._build_module_forward_from_graph(
            dcnnl_edges, dynapcnn_layers
        )

        # add extra edges marking which nodes are input to the network.
        self._add_entry_points_edges(dynapcnnlayers_handlers)

    def _add_entry_points_edges(self, dynapcnnlayers_handlers: dict) -> None:
        """Addes an extra edge `('input', X)` to `self.dcnnl_edges` if `X` is an entry point of the `DynapcnnNetwork`
        (i.e., `dynapcnnlayers_handlers[X]['layer_handler'].entry_point = True`).

        Parameters
        ----------
        - dynapcnnlayers_handlers (dict): a mapper containing `DynapcnnLayerHandler` instances along with their supporting metadata (e.g. assigned core,
            destination layers, etc.).
        """
        for indx, dcnnl_data in dynapcnnlayers_handlers.items():
            if dcnnl_data["layer_handler"].entry_point:
                self.dcnnl_edges.append(("input", indx))

    def _spot_merging_points(
        self, dcnnl_edges: list
    ) -> Dict[int, Dict[Tuple, sl.Merge]]:
        """Loops throught the edges of the computational graph from a `DynapcnnNetwork` to flag with nodes need
        input from a `Merge` layer and what the arguments of this layer should be.

        Parameters
        ----------
        - dcnnl_edges (list): tuples representing the output->input mapping between `DynapcnnLayer` instances
            that have been used as configuration for each core `CNNLayerConifg`.
        """

        nodes_with_merge_input = {}

        for edge in dcnnl_edges:
            trg_node = edge[1]
            fan_in = 0
            src_nodes = []

            # counts the fan-in for each target node `trg_node`.
            for edge_inner in dcnnl_edges:
                if edge_inner[1] == trg_node:
                    # fan-in update.
                    fan_in += 1
                    src_nodes.append(edge_inner[0])

            if fan_in == 2 and trg_node not in nodes_with_merge_input:
                # node needs input from a `Merge` layer: instantiate `Merge` and its arguments.
                nodes_with_merge_input[trg_node] = {
                    "sources": tuple(src_nodes),
                    "merge": sl.Merge(),
                }

            if fan_in > 2:
                raise ValueError(
                    f"Node {trg_node} is the has fan-in of {fan_in}: only fan-in of 2 is currently handled."
                )

        return nodes_with_merge_input

    def _build_module_forward_from_graph(
        self, dcnnl_edges: list, dynapcnn_layers: dict
    ) -> Union[Dict[int, DynapcnnLayer], Dict[Tuple, sl.Merge]]:
        """Creates two mappers, one indexing each `DynapcnnLayer` by its index (a node in `dcnnl_edges`) and another
        indexing the `DynapcnnLayer` instances (also by the index) that need their input being the output of a
        `Merge` layer (i.e., they are nodes in the graph where two different layer outputs converge to).

        Parameters
        ----------
        - dcnnl_edges (list): tuples representing the output->input mapping between `DynapcnnLayer` instances
            that have been used as configuration for each core `CNNLayerConifg`.
        - dynapcnn_layers (dict): a mapper containing `DynapcnnLayer` instances along with their supporting metadata (e.g. assigned core,
            destination layers, etc.).

        Returns
        ----------
        - forward_map (dict): a mapper where each `key` is the layer index (`DynapcnnLayer.dpcnnl_index`) and the `value` the layer instance itself.
        - merge_points (dict): a mapper where each `key` is the layer index and the `value` is a dictionary with a `Merge` layer (`merge_points[key]['merge'] = Merge()`,
            computing the input tensor to layer `key`) and its arguments (`merge_points[key]['sources'] = (int A, int B)`, where `A` and `B` are the `DynapcnnLayer`
            instances for which the ouput is to be used as the `Merge` arguments).
        """

        # mapper to flag nodes that need input from a `Merge` layer.
        merge_points = self._spot_merging_points(dcnnl_edges)

        # this dict. will be used to call the `forward` methods of each `DynapcnnLayer`.
        forward_map = {}

        for edge in dcnnl_edges:
            src_dcnnl = edge[0]  # source layer
            trg_dcnnl = edge[1]  # target layer

            if src_dcnnl not in forward_map:
                forward_map[src_dcnnl] = copy.deepcopy(
                    dynapcnn_layers[src_dcnnl]["layer"]
                )

            if trg_dcnnl not in forward_map:
                forward_map[trg_dcnnl] = copy.deepcopy(
                    dynapcnn_layers[trg_dcnnl]["layer"]
                )

        return forward_map, merge_points
