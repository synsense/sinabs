# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

from collections import defaultdict
import copy
from typing import Dict, List, Set, Tuple, Union
from warnings import warn

import torch.nn as nn
from torch import Tensor

import sinabs.layers as sl

from .dynapcnn_layer import DynapcnnLayer
from .dynapcnn_layer_handler import DynapcnnLayerHandler
from .utils import Edge, topological_sorting


class DynapcnnNetworkModule(nn.Module):
    """Allow forward (and backward) passing through a network of `DynapcnnLayer`s.

    Internally constructs a graph representation based on the provided
    `DynapcnnLayer` and `DynapcnnLayerHandler` instances and uses this
    to pass data through all layers in correct order.

    Parameters
    ----------
    - dynapcnn_layers (dict): a mapper containing `DynapcnnLayer` instances.
    - dynapcnnlayers_handlers (dict): a mapper containing `DynapcnnLayerHandler` instances

    Attributes
    ----------
    This class internally builds a graph with `DynapcnnLayer` as nodes and their
    connections as edges. Several data structures help efficient retrieval of
    information required for the forward pass:
    - `self._dynapcnnlayer_edges`: Set of edges connecting dynapcnn layers. Tuples
        of indices of source and target layers.
    - _sorted_nodes: List of layer indices in topological order, to ensure forward
        calls to layers only happen when required inputs are available.
    - _node_source_map: Dict with layer indices as keys and list of input layer indices
        as values.
    """

    def __init__(
        self,
        dynapcnn_layers: Dict[int, DynapcnnLayerHandler],
        dynapcnnlayers_handlers: Dict[int, DynapcnnLayerHandler],
    ):
        super().__init__()

        self._dynapcnn_layers = dynapcnn_layers
        self._dynapcnnlayer_handlers = dynapcnnlayers_handlers

    def setup_dynapcnnlayer_graph(self):
        """Set up data structures to run forward pass through dynapcnn layers"""
        self._dynapcnnlayer_edges = self.get_dynapcnnlayers_edges()
        self.add_entry_points_edges(self._dynapcnnlayer_edges)
        self._sorted_nodes = topological_sorting(self._dynapcnnlayer_edges)
        self._node_source_map = self.get_node_source_map(self._dynapcnnlayer_edges)
        # `Merge` layers are stateless. One instance can be used for all merge points.
        self._merge_layer = sl.Merge()

        # TODO: Probably not needed.
        # Collect layers with multiple inputs and instantiate `Merge` layers
        # self._merge_points = self._get_merging_points(self._node_source_map)

        # # create mappers to handle `DynapcnnLayer` instances' forward calling.
        # self.forward_map, self.merge_points = self._build_module_forward_from_graph(
        #     dcnnl_edges, dynapcnn_layers
        # )

    def get_dynapcnnlayers_edges(self) -> Set[Edge]:
        """Create edges representing connections between `DynapcnnLayer` instances.

        Returns
        ----------
        - dcnnl_edges: a set of edges using the IDs of `DynapcnnLayer` instances. These edges describe the computational
            graph implemented by the layers of the model (i.e., how the `DynapcnnLayer` instances address each other).
        """
        dcnnl_edges = set()

        for dcnnl_idx, handler in self._dynapcnnlayer_handlers.items():
            for dest in handler.destination_indices:
                dcnnl_edges.add((dcnnl_idx, dest))

        return dcnnl_edges

    def add_entry_points_edges(self, dcnnl_edges: Set[Edge]):
        """Add extra edges `('input', X)` to `dcnnl_edges` for
        layers which are entry points of the `DynapcnnNetwork`, i.e.
        `handler.entry_node = True`.

        Parameters
        ----------
        - dcnnl_edges (Set): tuples representing the output->input mapping between
        `DynapcnnLayer` instances. Will be changed in place.
        """
        for indx, handler in self._dynapcnnlayer_handlers.items():
            if handler.entry_node:
                dcnnl_edges.add(("input", indx))

    def get_node_source_map(self, dcnnl_edges: Set[Edge]) -> Dict[int, List[int]]:
        """From a set of edges, create a dict that maps to each node its sources

        Parameters
        ----------
        - dcnnl_edges (Set): tuples representing the output->input mapping between
        `DynapcnnLayer` instances.

        Returns
        -------
        - Dict with layer indices (int) as keys and list of layer indices that
            map to corresponding layer
        """
        sources = dict()

        for src, trg in dcnnl_edges:
            if trg in sources:
                sources[trg].append(src)
            else:
                sources[trg] = [src]

        return sources

    # TODO: Probably not needed
    def get_merging_points(
        self, node_source_map: Dict[int, List[int]]
    ) -> Dict[int, Dict[Tuple, sl.Merge]]:
        """Find nodes within `dcnnl_edges` that have multiple sources.

        Parameters
        ----------
        - node_source_map: Dict that maps to each layer index (int) a list of
            indices of layers that act as input source to this node

        Returns
        -------
        - Dict that for each layer with more than one input source maps its index
            (int) to a nested dict with two entries:
                * "sources": Set of indices of all source layers to this layer
                * "merge_layer": A `Merge` layer instance
        """
        return {
            tgt: {"sources": sources, "merge_layer": sl.Merge()}
            for tgt, sources in node_source_map
            if len(sources) > 1
        }

    def forward(
        self, x, return_complete: bool = False
    ) -> Union[Tensor, Dict[int, Dict[int, Tensor]]]:
        """Perform a forward pass through all dynapcnn layers
        The `setup_dynapcnnlayer_graph` method has to be executed beforehand.

        Parameters
        ----------
        x: Tensor that serves as input to network. Is passed to all layers
            that are marked as entry points
        return_complete: bool that indicates whether all layer outputs should
            be return or only those with no further destinations (default)

        Returns
        -------
        The returned object depends on whether `return_complete` is set and on
        the network configuration:
        * If `return_complete` is `True`, all layer outputs will be returned in a
            dict, with layer indices as keys, and nested dicts as values, which
            hold destination indices as keys and output tensors as values.
        * If `return_complete` is `False` and there is only a single destination
            in the whole network that is marked as final (i.e. destination
            index in dynapcnn layer handler is negative), it will return the
            output as a single tensor.
        * If `return_complete` is `False` and no destination in the network
            is marked as final, a warning will be raised and the function
            returns an empty dict.
        * In all other cases a dict will be returned that is of the same
            structure as if `return_complete` is `True`, but only with entries
            where the destination is marked as final.

        """
        if not hasattr(self, "_sorted_nodes"):
            raise RuntimeError(
                "It looks like `setup_dynapcnnlayers_graph` has never been executed. "
                "It needs to be called at least once before calling `forward`."
            )

        # For each layer store its outputs as dict with destination layers as keys.
        # For input use `defaultdict` so it can be used for all destinations where needed
        layers_outputs = {"input": defaultdict(lambda: x)}

        for idx_curr in self._sorted_nodes:
            # Get inputs to the layer
            if len(sources := self._node_source_map[idx_curr]) > 1:
                # Layer has multiple inputs
                inputs = [layers_outputs[idx_src][idx_curr] for idx_src in sources]
                current_input = self._merge_layer(*inputs)
            else:
                idx_src = sources[0]
                current_input = layers_outputs[idx_src][idx_curr]

            # Get current layer instance and destinations
            layer = self._dynapcnn_layers[idx_curr]
            destinations = self._dynapcnnlayer_handlers[idx_curr].destination_indices

            # Forward pass through layer
            output = layer(current_input)

            # Store layer output for all destinations
            if len(destinations) == 1:
                # Output is single tensor
                layers_outputs[idx_curr] = {destinations[0]: output}
            else:
                # Output is list of tensors for different destinations
                layers_outputs[idx_curr] = {
                    idx_dest: out for idx_dest, out in zip(destinations, output)
                }

        if return_complete:
            return layers_outputs

        # Take outputs with final destinations as network output
        network_outputs = {}
        for layer_idx, outputs in layers_outputs.items():
            final_outputs = {
                abs(idx_dest): out for idx_dest, out in outputs.items() if idx_dest < 0
            }
            if final_outputs:
                network_outputs[layer_idx] = final_outputs

        # If no outputs have been found return None and warn
        if not network_outputs:
            warn(
                "No final outputs have been found. Try setting `return_complete` "
                "`True` to get all outputs, or mark final outputs by setting "
                "corresponding destination layer indices in DynapcnnLayerHandler "
                " instance to negative integer values"
            )
            return

        # Special case with single output: return single tensor
        if (
            len(network_outputs) == 1
            and len(out := (next(iter(network_outputs.values())))) == 1
        ):
            return out

        # If there is output from multiple layers return all of them in a dict
        return network_outputs

    # TODO: Necessary?
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

        return forward_map
