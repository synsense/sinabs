# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

import copy
from typing import Dict, List, Tuple, Type, Set

import nirtorch
import torch
import torch.nn as nn

import sinabs

from .utils import topological_sorting


class GraphExtractor:
    def __init__(self, spiking_model: nn.Module, dummy_input: torch.tensor):
        """Class implementing the extraction of the computational graph from `spiking_model`, where
        each node represents a layer in the model and the list of edges represents how the data flow between
        the layers.

        Parameters
        ----------
        - spiking_model (nn.Module): a sinabs-compatible spiking network.
        - dummy_input (torch.tensor): a random input sample to be fed through the model to acquire both
            the computational graph (via `nirtorch`) and the I/O shapes of each node. Its a 4-D shape
            with `(batch, channels, heigh, width)`.

        Attributes
        ----------
        - edges (set of 2-tuples of integers):
            Tuples describing the connections between layers in `spiking_model`.
            Each layer (node) is identified by a unique integer ID.
        - name_2_index_map (dict):
            Keys are original variable names of layers in `spiking_model`.
            Values are unique integer IDs.
        - entry_nodes (set of ints):
            IDs of nodes acting as entry points for the network, i.e. receiving external input.
        - modules_map (dict):
            Map from layer ID to the corresponding nn.Module instance.
        """

        # extract computational graph.
        nir_graph = nirtorch.extract_torch_graph(
            spiking_model, dummy_input, model_name=None
        ).ignore_tensors()

        # converts the NIR representation into a list of edges with nodes represented as integers.
        self._edges, self._name_2_indx_map, self._entry_nodes = (
            self._get_edges_from_nir(nir_graph)
        )

        # recovers the associated `nn.Module` (layer) of each node.
        self._modules_map = self._get_named_modules(spiking_model)

        # retrieves what the I/O shape for each node's module is.
        self._nodes_io_shapes = self._get_nodes_io_shapes(dummy_input)

    ####################################################### Publich Methods #######################################################

    @property
    def entry_nodes(self) -> Set[int]:
        return {n for n in self._entry_nodes}

    @property
    def edges(self) -> Set[Tuple[int, int]]:
        return {(src, tgt) for src, tgt in self._edges}

    @property
    def name_2_indx_map(self) -> Dict[str, int]:
        return {name: idx for name, idx in self._name_2_indx_map.items()}

    @property
    def nodes_io_shapes(self) -> Dict[int, torch.Size]:
        return {n: size for n, size in self._nodes_io_shapes.items()}

    @property
    def sorted_nodes(self) -> List[int]:
        return [n for n in self._sort_graph_nodes()]

    @property
    def modules_map(self) -> Dict[int, nn.Module]:
        return {n: module for n, module in self._modules_map.items()}

    def remove_nodes_by_class(
        self, node_classes: Tuple[Type]
    ) -> Tuple[Set[int], Dict[int, int]]:
        """Remove nodes of given classes from graph in place.

        Create a new set of edges, considering layers that `DynapcnnNetwork` will ignore. This
        is done by setting the source (target) node of an edge where the source (target) node
        will be dropped as the node that originally targeted this node to be dropped.

        Will change internal attributes `self._edges`, `self._entry_nodes`,
        `self._name_2_indx_map`, and `self._nodes_io_shapes` to reflect the changes.

        Parameters
        ----------
        - node_classes (tuple of types):
            Layer classes that should be removed from the graph.

        """
        # Compose new graph by creating a dict with all remaining node IDs as keys and set of target node IDs as values
        source2target: Dict[int, Set[int]] = {
            node: self._find_valid_targets(node, node_classes)
            for node in self.sorted_nodes
            # Skip nodes that are to be removed
            if not isinstance(self.modules_map[node], node_classes)
        }

        # remapping nodes indices contiguously starting from 0
        remapped_nodes = {
            old_idx: new_idx
            for new_idx, old_idx in enumerate(sorted(source2target.keys()))
        }

        # Parse new set of edges based on remapped node IDs
        self._edges = {
            (remapped_nodes[src], remapped_nodes[tgt])
            for src, targets in source2target.items()
            for tgt in targets
        }

        # Update name-to-index map based on new node indices
        self._name_2_indx_map = {
            name: remapped_nodes[old_idx]
            for name, old_idx in self._name_2_indx_map.items()
            if old_idx in remapped_nodes
        }

        # Update entry nodes based on new node indices
        self._entry_nodes = {
            remapped_nodes[old_idx]
            for old_idx in self._entry_nodes
            if old_idx in remapped_nodes
        }

        # Update io-shapes based on new node indices
        self._nodes_io_shapes = {
            remapped_nodes[old_idx]: shape
            for old_idx, shape in self._nodes_io_shapes.items()
            if old_idx in remapped_nodes
        }

        # Update sinabs module map based on new node indices
        self._modules_map = {
            remapped_nodes[old_idx]: module
            for old_idx, module in self._modules_map.items()
            if old_idx in remapped_nodes
        }

    def get_node_io_shapes(self, node: int) -> Tuple[torch.Size, torch.Size]:
        """Returns the I/O tensors' shapes of `node`.

        Returns
        ----------
        - input shape (torch.Size): shape of the input tensor to `node`.
        - output shape (torch.Size): shape of the output tensor from `node`.
        """
        return (
            self._nodes_io_shapes[node]["input"],
            self._nodes_io_shapes[node]["output"],
        )

    ####################################################### Pivate Methods #######################################################

    def _get_edges_from_nir(
        self, nir_graph: nirtorch.graph.Graph
    ) -> Tuple[List[Tuple[int, int]], Dict[str, int], List[int]]:
        """Standardize the representation of `nirtorch.graph.Graph` into a list of edges (`Tuple[int, int]`) where each node in `nir_graph` is represented by an interger (with the source node starting as `0`).

        Parameters
        ----------
        - nir_graph (nirtorch.graph.Graph): a NIR graph representation of `spiking_model`.

        Returns
        ----------
        - edges (set): tuples describing the connections between layers in `spiking_model`.
        - name_2_indx_map (dict): `key` is the original variable name for a layer in `spiking_model` and `value is an integer representing the layer in a standard format.
        - entry_nodes (set): IDs of nodes acting as entry points for the network (i.e., receiving external input).
        """
        edges = set()
        name_2_indx_map = {}

        # TODO maybe make sure the input node from nir always gets assined `0`.

        for src_node in nir_graph.node_list:
            # Make sure current node is in `name_2_indx_map`
            if src_node.name not in name_2_indx_map:
                # Assign unique index by taking current length of `name_2_indx_map`
                name_2_indx_map[src_node.name] = len(name_2_indx_map)

            for trg_node in src_node.outgoing_nodes:
                # Make sure all targets of current node are in `name_2_indx_map`
                if trg_node.name not in name_2_indx_map:
                    name_2_indx_map[trg_node.name] = len(name_2_indx_map)

                # Store the edge of current node to the target
                edges.add(
                    (name_2_indx_map[src_node.name], name_2_indx_map[trg_node.name])
                )

        # finding entry/exits nodes of the graph.
        all_sources, all_targets = zip(*edges)

        entry_nodes = set(all_sources) - set(all_targets)

        return edges, name_2_indx_map, entry_nodes

    def _get_named_modules(self, model: nn.Module) -> Dict[int, nn.Module]:
        """Find for each node in the graph what its associated layer in `model` is.

        Parameters
        ----------
        - model (nn.Module): the `spiking_model` used as argument to the class instance.

        Returns
        ----------
        - modules_map (dict): the mapping between a node (`key` as an `int`) and its module (`value` as a `nn.Module`).
        """
        modules_map = {}

        if isinstance(
            model, nn.Sequential
        ):  # TODO shouldn't accept `nn.Sequential` any longer.
            # access modules via `.named_modules()`.
            for name, module in model.named_modules():
                if name != "":
                    # skip the module itself.
                    modules_map[self._name_2_indx_map[name]] = module

        elif isinstance(model, nn.Module):
            # access modules via `.named_children()`.
            for name, module in model.named_children():
                modules_map[self._name_2_indx_map[name]] = module

        else:
            raise ValueError("Either a nn.Sequential or a nn.Module is required.")

        return modules_map

    def _sort_graph_nodes(self) -> List[int]:
        """Sort graph nodes topologically.

        Returns
        -------
        - sorted_nodes (list of integers): IDs of nodes, sorted.
        """
        # Make a temporary copy of edges and include inputs
        temp_edges = self.edges
        for node in self._entry_nodes:
            temp_edges.add(("input", node))
        return topological_sorting(temp_edges)

    def _get_nodes_io_shapes(
        self, input_dummy: torch.tensor
    ) -> Dict[int, Dict[str, torch.Size]]:
        """Iteratively calls the forward method of each `nn.Module` (i.e., a layer/node in the graph) using the topologically
        sorted nodes extracted from the computational graph of the model being parsed.

        Parameters
        ----------
        - input_dummy (torch.tensor): a sample (random) tensor of the sort of input being fed to the network.

        Returns
        ----------
        - nodes_io_map (dict): a dictionary mapping nodes to their I/O shapes.
        """
        nodes_io_map = {}

        # propagate inputs through the nodes.
        for node in self.sorted_nodes:

            if isinstance(self.modules_map[node], sinabs.layers.merge.Merge):
                # find `Merge` arguments (at this point the inputs to Merge should have been calculated).
                arg1, arg2 = self._find_merge_arguments(node)

                # retrieve arguments output tensors.
                arg1_out = nodes_io_map[arg1]["output"]
                arg2_out = nodes_io_map[arg2]["output"]

                # TODO - this is currently a limitation imposed by the validation checks done by Speck once a configuration: it wants two
                # different input sources to a core to have the same output shapes.
                if arg1_out.shape != arg2_out.shape:
                    raise ValueError(
                        f"Layer `sinabs.layers.merge.Merge` (node {node}) require two input tensors with the same shape: arg1.shape {arg1_out.shape} differs from arg2.shape {arg2_out.shape}."
                    )

                # forward input through the node.
                _output = self.modules_map[node](arg1_out, arg2_out)

                # save node's I/O tensors.
                nodes_io_map[node] = {"input": arg1_out, "output": _output}

            else:

                if node in self._entry_nodes:
                    # forward input dummy through node.
                    _output = self.modules_map[node](input_dummy)

                    # save node's I/O tensors.
                    nodes_io_map[node] = {"input": input_dummy, "output": _output}

                else:
                    # find node generating the input to be used.
                    input_node = self._find_source_of_input_to(node)
                    _input = nodes_io_map[input_node]["output"]

                    # forward input through the node.
                    _output = self.modules_map[node](_input)

                    # save node's I/O tensors.
                    nodes_io_map[node] = {"input": _input, "output": _output}

        # replace the I/O tensor information by its shape information.
        for node, io in nodes_io_map.items():
            nodes_io_map[node]["input"] = io["input"].shape
            nodes_io_map[node]["output"] = io["output"].shape

        return nodes_io_map

    def _find_all_sources_of_input_to(self, node: int) -> Set[int]:
        """Finds all source nodes to `node`.

        Parameters
        ----------
        - node (int): the node in the computational graph for which we whish to find the input source (either another node in the
            graph or the original input itself to the network).

        Returns
        ----------
        - input sources (set of int): IDs of the nodes in the computational graph providing the input to `node`.
        """
        return set(src for (src, tgt) in self._edges if tgt == node)

    def _find_source_of_input_to(self, node: int) -> int:
        """Finds the first edge `(X, node)` returns `X`.

        Parameters
        ----------
        - node (int): the node in the computational graph for which we whish to find the input source (either another node in the
            graph or the original input itself to the network).

        Returns
        ----------
        - input source (int): ID of the node in the computational graph providing the input to `node`. If `node` is
            receiving outside input (i.e., it is a starting node) the return will be -1. For example, this will be the case
            when a network with two independent branches (each starts from a different "input node") merge along the computational graph.
        """
        sources = self._find_all_sources_of_input_to(node)
        if len(sources) == 0:
            return -1
        if len(sources) > 1:
            raise RuntimeError(f"Node {node} has more than 1 input")
        return sources.pop()

    def _find_merge_arguments(self, node: int) -> Tuple[int, int]:
        """A `Merge` layer receives two inputs. Return the two inputs to `merge_node` representing a `Merge` layer.

        Returns
        ----------
        - args (tuple): the IDs of the nodes that provice the input arguments to a `Merge` layer.
        """
        sources = self._find_all_sources_of_input_to(node)

        if len(sources) != 2:
            raise ValueError(
                f"Number of arguments found for `Merge` node {merge_node} is {len(args)} (should be 2)."
            )

        return tuple(sources)

    def _find_valid_targets(
        self, node: int, ignored_node_classes: Tuple[Type]
    ) -> Set[int]:
        """Find all targets of a node that are not ignored classes

        Return a set of all target nodes that are not of an ignored class.
        For target nodes of ignored classes, recursively return their valid
        targets.

        Parameters
        ----------
        - node (int): ID of node whose targets should be found
        - ignored_node_classes (tuple of types): Classes of which nodes should be skiped

        Returns
        -------
        - valid_targets (set of int): Set of all recursively found target IDs
        """
        targets = set()
        for src, tgt in self.edges:
            # Search for all edges with node as source
            if src == node:
                if isinstance(self.modules_map[tgt], ignored_node_classes):
                    # Find valid targets of target
                    targets.update(self._find_valid_targets(tgt, ignored_node_classes))
                else:
                    # Target is valid, add it to `targets`
                    targets.add(tgt)
        return targets
