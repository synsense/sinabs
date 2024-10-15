# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

from collections import defaultdict
from typing import Dict, List, Set, Union
from warnings import warn

import torch.nn as nn
from torch import Tensor

import sinabs.layers as sl

from .dynapcnn_layer import DynapcnnLayer
from .utils import Edge, topological_sorting


class DynapcnnNetworkModule(nn.Module):
    """Allow forward (and backward) passing through a network of `DynapcnnLayer`s.

    Internally constructs a graph representation based on the provided arguments
    and uses this to pass data through all layers in correct order.

    Parameters
    ----------
    - dynapcnn_layers (dict): a mapper containing `DynapcnnLayer` instances.
    - destination_map (dict): Maps layer indices to list of destination indices.
        Exit destinations are marked by negative integers    
    - entry_points (set): Set of layer indices that act as network entry points

    Attributes
    ----------
    This class internally builds a graph with `DynapcnnLayer` as nodes and their
    connections as edges. Several data structures help efficient retrieval of
    information required for the forward pass:
    - _dynapcnnlayer_edges: Set of edges connecting dynapcnn layers. Tuples
        of indices of source and target layers.
    - _sorted_nodes: List of layer indices in topological order, to ensure forward
        calls to layers only happen when required inputs are available.
    - _node_source_map: Dict with layer indices as keys and list of input layer indices
        as values.
    """

    def __init__(
        self,
        dynapcnn_layers: Dict[int, DynapcnnLayer],
        destination_map: Dict[int, List[int]],
        entry_points: Set[int],
    ):
        super().__init__()

        self.dynapcnn_layers = dynapcnn_layers
        self._destination_map = destination_map
        self._entry_points = entry_points

        # `Merge` layers are stateless. One instance can be used for all merge points during forward pass
        self._merge_layer = sl.Merge()
    
    @property
    def destination_map(self):
        return self._destination_map
    
    @property
    def entry_points(self):
        return self._entry_points

    @property
    def sorted_nodes(self):
        return self._sorted_nodes
    
    @property
    def node_source_map(self):
        return self._node_source_map

    def setup_dynapcnnlayer_graph(self, index_layers_topologically: bool = False):
        """ Set up data structures to run forward pass through dynapcnn layers
        
        Parameters
        ----------
        - index_layers_topologically (bool): If True, will assign new indices to
            dynapcnn layers such that they match their topological order within the
            network graph. This is not necessary but can help understand the network
            more easily when inspecting it.
        """
        self._dynapcnnlayer_edges = self.get_dynapcnnlayers_edges()
        self.add_entry_points_edges(self._dynapcnnlayer_edges)
        self._sorted_nodes = topological_sorting(self._dynapcnnlayer_edges)
        self._node_source_map = self.get_node_source_map(self._dynapcnnlayer_edges)
        if index_layers_topologically:
            self.reindex_layers(self._sorted_nodes)

    def get_dynapcnnlayers_edges(self) -> Set[Edge]:
        """Create edges representing connections between `DynapcnnLayer` instances.

        Returns
        ----------
        - dcnnl_edges: a set of edges using the IDs of `DynapcnnLayer` instances. These edges describe the computational
            graph implemented by the layers of the model (i.e., how the `DynapcnnLayer` instances address each other).
        """
        dcnnl_edges = set()

        for dcnnl_idx, destination_indices in self._destination_map.items():
            for dest in destination_indices:
                if dest >= 0:  # Ignore negative destinations (network exit points)
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
        for indx in self._entry_points:
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
            layer = self.dynapcnn_layers[idx_curr]
            destinations = self._destination_map[idx_curr]

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
                "corresponding destination layer indices in destination_map "
                " to negative integer values"
            )
            return

        # Special case with single output: return single tensor
        if (
            len(network_outputs) == 1
            and len(out := (next(iter(network_outputs.values())))) == 1
        ):
            return next(iter(out.values()))

        # If there is output from multiple layers return all of them in a dict
        return network_outputs

    def reindex_layers(self, index_order: List[int]):
        """ Reindex layers based on provided order
        
        Will assign new index to dynapcnn layers and update all internal
        attributes accordingly.

        Parameters
        ----------
        index_order: List of integers indicating new order of layers:
            Position of layer index within this list indicates new index
        """
        def negative_default(key):
            if isinstance(key, int) and key < 0:
                return key
            else:
                raise KeyError(key)

        mapping = {old: new for new, old in enumerate(index_order)}

        def remap(key):
            if key == "input":
                return "input"
            if isinstance(key, int) and key < 0:
                # maintain negative indices
                return key
            else:
                return mapping[key]

        # Remap all internal objects
        self.dynapcnn_layers = {remap(idx): lyr for idx, lyr in self.dynapcnn_layers.items()}
        self._entry_points = {remap(idx) for idx in self._entry_points}
        self._destination_map = {
            remap(idx): [remap(dest) for dest in destinations]
            for idx, destinations in self._destination_map.items()
        }
        self._dynapcnnlayer_edges = {
            (remap(src), remap(trg)) for (src, trg) in self._dynapcnnlayer_edges
        }
        self._sorted_nodes = [remap(idx) for idx in self._sorted_nodes]
        self._node_source_map = {
            remap(node): [remap(src) for src in sources]
            for node, sources in self._node_source_map.items()
        }