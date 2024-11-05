# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union

import nirtorch
import torch
import torch.nn as nn

import sinabs
from .dvs_layer import DVSLayer

from .connectivity_specs import (
    LAYER_TYPES_WITH_MULTIPLE_INPUTS,
    LAYER_TYPES_WITH_MULTIPLE_OUTPUTS,
)
from .dynapcnn_layer_utils import construct_dynapcnnlayers_from_mapper
from .dynapcnnnetwork_module import DynapcnnNetworkModule
from .exceptions import InvalidGraphStructure, InvalidModelWithDVSSetup
from .sinabs_edges_handler import collect_dynapcnn_layer_info, get_dvs_node_from_mapper, fix_dvs_module_edges, merge_dvs_pooling_edge
from .utils import Edge, topological_sorting

try:
    from nirtorch.graph import TorchGraph
except ImportError:
    # In older nirtorch versions TorchGraph is called Graph
    from nirtorch.graph import Graph as TorchGraph


class GraphExtractor:
    def __init__(
        self,
        spiking_model: nn.Module,
        dummy_input: torch.tensor,
        dvs_input: Optional[bool] = None,
        ignore_node_types: Optional[Iterable[Type]] = None,
    ):
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
        - indx_2_module_map (dict):
            Map from layer ID to the corresponding nn.Module instance.
        - nodes_io_shapes (dict):
            Map from node ID to dict containing node's in- and output shapes
        - dvs_input (bool): optional (default as `None`). Whether or not the model 
            should start with a `DVSLayer`.
        - ignore_node_types (iterable of types): Node types that should be
            ignored completely from the graph. This can include, for instance,
            `nn.Dropout2d`, which otherwise can result in wrongly inferred
            graph structures by NIRTorch. Types such as `nn.Flatten`, or sinabs
            `Merge` should not be included here, as they are needed to properly
            handle graph structure and metadata. They can be removed after
            instantiation with `remove_nodes_by_class`.
        """

        # extract computational graph.
        nir_graph = nirtorch.extract_torch_graph(
            spiking_model, dummy_input, model_name=None
        ).ignore_tensors()
        if ignore_node_types is not None:
            for node_type in ignore_node_types:
                nir_graph = nir_graph.ignore_nodes(node_type)

        # Map node names to indices
        self._name_2_indx_map = self._get_name_2_indx_map(nir_graph)

        # Extract edges list from graph
        self._edges = self._get_edges_from_nir(nir_graph, self._name_2_indx_map)

        # Store the associated `nn.Module` (layer) of each node.
        self._indx_2_module_map = self._get_named_modules(spiking_model)

        # Determine entry points to graph
        self._entry_nodes = self._get_entry_nodes(self._edges)

        # Make sure DVS input is properly integrated into graph
        self._handle_dvs_input(input_shape=dummy_input.shape[1:], dvs_input=dvs_input)

        # Verify that graph is compatible
        self.verify_graph_integrity()

        # retrieves what the I/O shape for each node's module is.
        self._nodes_io_shapes = self._get_nodes_io_shapes(dummy_input)

    ####################################################### Publich Methods #######################################################

    @property
    def dvs_layer(self) -> Set[int]:
        return {n for n in self._get_dvs_layer}

    @property
    def entry_nodes(self) -> Set[int]:
        return {n for n in self._entry_nodes}

    @property
    def edges(self) -> Set[Edge]:
        return {(src, tgt) for src, tgt in self._edges}

    @property
    def has_dvs_layer(self) -> Set[Edge]:
        return self.dvs_layer is not None

    @property
    def name_2_indx_map(self) -> Dict[str, int]:
        return {name: idx for name, idx in self._name_2_indx_map.items()}

    @property
    def nodes_io_shapes(self) -> Dict[int, Tuple[torch.Size]]:
        return {n: size for n, size in self._nodes_io_shapes.items()}

    @property
    def sorted_nodes(self) -> List[int]:
        return [n for n in self._sort_graph_nodes()]

    @property
    def indx_2_module_map(self) -> Dict[int, nn.Module]:
        return {n: module for n, module in self._indx_2_module_map.items()}

    def get_dynapcnn_network_module(
        self, discretize: bool = False, weight_rescaling_fn: Optional[Callable] = None
    ) -> DynapcnnNetworkModule:
        """Create DynapcnnNetworkModule based on stored graph representation

        This includes construction of the DynapcnnLayer instances

        Parameters:
        -----------
        - discretize (bool): If `True`, discretize the parameters and thresholds. This is needed for uploading
            weights to dynapcnn. Set to `False` only for testing purposes.
        - weight_rescaling_fn (callable): a method that handles how the re-scaling factor for one or more `SumPool2d` projecting to
            the same convolutional layer are combined/re-scaled before applying them.

        Returns
        -------
        - The DynapcnnNetworkModule based on graph representation of this `GraphExtractor`

        """
        # create a dict holding the data necessary to instantiate a `DynapcnnLayer`.
        self.dcnnl_map = collect_dynapcnn_layer_info(
            indx_2_module_map=self.indx_2_module_map,
            edges=self.edges,
            nodes_io_shapes=self.nodes_io_shapes,
            entry_nodes=self.entry_nodes,
        )

        # build `DynapcnnLayer` instances from mapper.
        dynapcnn_layers, destination_map, entry_points = (
            construct_dynapcnnlayers_from_mapper(
                dcnnl_map=self.dcnnl_map,
                discretize=discretize,
                rescale_fn=weight_rescaling_fn,
            )
        )

        # DVSLayer node information (None if DVS camera is not used).
        dvs_node_info = self.dcnnl_map.get("dvs", None)

        # Instantiate the DynapcnnNetworkModule
        return DynapcnnNetworkModule(
            dynapcnn_layers, destination_map, entry_points, dvs_node_info
        )

    def remove_nodes_by_class(self, node_classes: Tuple[Type]):
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
        source2target: Dict[int, Set[int]] = {}
        for node in self.sorted_nodes:
            if isinstance((mod := self.indx_2_module_map[node]), node_classes):
                # If an entry node is removed, its targets become entry nodes
                if node in self.entry_nodes:
                    targets = self._find_valid_targets(node, node_classes)
                    self._entry_nodes.update(targets)

                # Update input shapes of nodes after `Flatten` to the shape before flattening
                # Note: This is likely to produce incorrect results if multiple Flatten layers
                # come in sequence.
                if isinstance(mod, nn.Flatten):
                    shape_before_flatten = self.nodes_io_shapes[node]["input"]
                    for target_node in self._find_valid_targets(node, node_classes):
                        self._nodes_io_shapes[target_node][
                            "input"
                        ] = shape_before_flatten

            else:
                source2target[node] = self._find_valid_targets(node, node_classes)

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

        # Update internal graph representation according to changes
        self._update_internal_representation(remapped_nodes)

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

    def verify_graph_integrity(self):
        """Apply checks to verify that graph is supported

        Currently this checks that only nodes of specific classes have
        multiple sources or targets. This method might be extended in the
        future to implement stricter formal verification.
        """
        # Iterate over all nodes, and count its sources and targets
        for node, module in self.indx_2_module_map.items():
            # Check sources
            if not isinstance(module, LAYER_TYPES_WITH_MULTIPLE_INPUTS):
                sources = self._find_all_sources_of_input_to(node)
                if len(sources) > 1:
                    raise InvalidGraphStructure(
                        f"Only nodes of type {LAYER_TYPES_WITH_MULTIPLE_INPUTS} "
                        f"can have more than one input. Node {node} is of type "
                        f"{type(module)} and has {len(sources)} inputs."
                    )
            # Check targets
            if not isinstance(module, LAYER_TYPES_WITH_MULTIPLE_OUTPUTS):
                targets = self._find_valid_targets(node)
                if len(targets) > 1:
                    raise InvalidGraphStructure(
                        f"Only nodes of type {LAYER_TYPES_WITH_MULTIPLE_OUTPUTS} "
                        f"can have more than one output. Node {node} is of type "
                        f"{type(module)} and has {len(targets)} outputs."
                    )

    ####################################################### Pivate Methods #######################################################

    def _handle_dvs_input(self, input_shape: Tuple[int, int, int], dvs_input: Optional[bool] = None):
        """Make sure DVS input is properly integrated into graph
        
        - Decide whether `DVSLayer` instance needs to be added to the graph
            This is the case when `dvs_input==True` and there is no `DVSLayer` yet.
        - Make sure edges between DVS related nodes are set properly
        - Absorb pooling layers in DVS node if applicable
        
        Parameters
        ----------
        - input_shape (tuple of three integers): Input shape (features, height, width)
        - dvs_input (bool or `None` (default)): If `False`, will raise
            `InvalidModelWithDvsSetup` if a `DVSLayer` is part of the graph. If `True`,
            a `DVSLayer` will be added to the graph if there is none already. If `None`,
            the model is considered to be using DVS input only if the graph contains
            a `DVSLayer`.
        """
        # If DVS camera is wanted but `spiking_model` does not start with DVS layer.
        if self._need_dvs_node(dvs_input):
            # Insert a DVSLayer node in the graph.
            self._add_dvs_node(dvs_input_shape=input_shape)
        
        # Check for the need of fixing NIR edges extraction when DVS is a node in the graph. If DVS
        # is used its node becomes the only entry node in the graph.
        fix_dvs_module_edges(self._edges, self._indx_2_module_map, self._name_2_indx_map, self._entry_nodes)

        # Merge a pooling node from a 'dvs-pooling' edge (pooling being an independent node in the original 
        # graph) into the DVSLayer if such edge exists.
        merge_dvs_pooling_edge(self._edges, self._indx_2_module_map, self._name_2_indx_map)

        # Check if graph structure and DVSLayer.merge_polarities are correctly set (if DVS node exists).
        self._validate_dvs_setup(dvs_input_shape=input_shape)

    def _add_dvs_node(self, dvs_input_shape: Tuple[int, int, int]) -> None:
        """ In-place modification of `self._name_2_indx_map`, `self._indx_2_module_map`, and `self._edges` to accomodate the 
        creation of an extra node in the graph representing the DVS camera of the chip. The DVSLayer node will point to every
        other node that is up to this point an entry node of the original graph, so `self._entry_nodes` is modified in-place
        to have only one entry: the index of the DVS node.

        Parameters
        ----------
        - dvs_input_shape (tuple): shape of the DVSLayer input in format `(features, height, width)`.
        """

        (features, height, width) = dvs_input_shape
        if features > 2:
            raise ValueError(f'A DVSLayer istance can have the feature dimension of its inputs with values 1 or 2 but {features} was given.')

        # add name entry for node 'dvs'.
        self._name_2_indx_map['dvs'] = len(self._name_2_indx_map)
        # add module entry for node 'dvs'.
        self._indx_2_module_map[self._name_2_indx_map['dvs']] = DVSLayer(
            input_shape=(height, width),
            merge_polarities=(features == 1),
        )
        # set DVS node as input to each entry node of the graph.
        self._edges.update({(self._name_2_indx_map['dvs'], entry_node) for entry_node in self._entry_nodes})
        # DVSLayer node becomes the only entrypoint of the graph.
        self._entry_nodes = {self._name_2_indx_map['dvs']}

    def _need_dvs_node(self, dvs_input: Optional[bool] = None) -> bool:
        """ Returns whether or not a node will need to be added to represent a
        `DVSLayer` instance. 
        
        A new node will have to be added if `self._indx_2_module_map` contains no
        `DVSLayer` instance and `dvs_input == True`.
        
        Parameters
        ----------
        - dvs_input (bool): optional (default as `None`). Wether or not dynapcnn receive input from its DVS camera.
        Returns
        -------
        - True if `self._indx_2_module_map` contains a DVSLayer, False otherwise.
        """

        has_dvs_layer = self.has_dvs_layer

        # Checks if DVSLayer instance exists but user has set 'dvs_input' to False.
        if has_dvs_layer and dvs_input == False:  # `== False` ensures that `None` is not considered
            raise InvalidModelWithDVSSetup()

        return not has_dvs_layer and dvs_input
    
    def _get_dvs_layer(self) -> Union[DVSLayer, None]:
        """ Loops though all modules and return `DVSLayer` instance if it exists.
        
        Returns
        -------
        - DVSLayer if exactly one is found, otherwise None

        Raises
        ------
        - InvalidGraphStructure if more than one DVSLayer is found

        """

        dvs_layers = {
            module for module in self._indx_2_module_map.values()
            if isinstance(module, DVSLayer)
        }

        if (num_dvs := len(dvs_layers)) == 0:
            return
        elif num_dvs == 1:
            return dvs_layers[0]
        else:
            raise InvalidGraphStructure(
                f"The provided model has {num_dvs} `DVSLayer`s. At most one is allowed."
            )
        
    def _validate_dvs_setup(self, dvs_input_shape: Tuple[int, int, int]) -> None:
        """ If a DVSLayer node exists, makes sure it is the only entry node of the graph. Checks if its `merge_polarities`
        attribute matches `dummy_input.shape[0]` (the number of features) and, if not, it will be set based on the numeber of
        features of the input.

        Parameters
        ----------
        - dvs_input_shape (tuple): shape of the DVSLayer input in format `(features, height, width)`.
        """

        dvs_layer = self.dvs_layer

        if (nb_entries := len(self._entry_nodes)) > 1:
            raise ValueError(f'A DVSLayer node exists and there are {nb_entries} entry nodes in the graph: the DVSLayer should be the only entry node.')
        
        (features, _, _) = dvs_input_shape

        if features > 2:
            raise ValueError(f'A DVSLayer istance can have the feature dimension of its inputs with values 1 or 2 but {features} was given.')
        
        if dvs_layer.merge_polarities and features != 1:
            raise ValueError(f"The 'DVSLayer.merge_polarities' is set to 'True' which means the number of input features should be 1 (current input shape is {dvs_input_shape}).")
            
        if features == 1:
            dvs_layer.merge_polarities = True

    def _get_name_2_indx_map(self, nir_graph: TorchGraph) -> Dict[str, int]:
        """Assign unique index to each node and return mapper from name to index.

        Parameters
        ----------
        - nir_graph (TorchGraph): a NIR graph representation of `spiking_model`.

        Returns
        ----------
        - name_2_indx_map (dict): `key` is the original variable name for a layer in
            `spiking_model` and `value is an integer representing the layer in a standard format.
        """

        return {
            node.name: node_idx for node_idx, node in enumerate(nir_graph.node_list)
        }

    def _get_edges_from_nir(
        self, nir_graph: TorchGraph, name_2_indx_map: Dict[str, int]
    ) -> Set[Edge]:
        """Standardize the representation of TorchGraph` into a list of edges,
        representing nodes by their indices.

        Parameters
        ----------
        - nir_graph (TorchGraph): a NIR graph representation of `spiking_model`.
        - name_2_indx_map (dict): Map from node names to unique indices.

        Returns
        ----------
        - edges (set): tuples describing the connections between layers in `spiking_model`.
        - name_2_indx_map (dict): `key` is the original variable name for a layer in `spiking_model` and `value is an integer representing the layer in a standard format.
        - entry_nodes (set): IDs of nodes acting as entry points for the network (i.e., receiving external input).
        """
        return {
            (name_2_indx_map[src.name], name_2_indx_map[tgt.name])
            for src in nir_graph.node_list
            for tgt in src.outgoing_nodes
        }

    def _get_entry_nodes(self, edges: Set[Edge]) -> Set[Edge]:
        """Find nodes that act as entry points to the graph

        Parameters
        ----------
        - edges (set): tuples describing the connections between layers in `spiking_model`.

        Returns
        ----------
        - entry_nodes (set): IDs of nodes acting as entry points for the network
           (i.e., receiving external input).
        """
        all_sources, all_targets = zip(*edges)
        return set(all_sources) - set(all_targets)

    def _get_named_modules(self, model: nn.Module) -> Dict[int, nn.Module]:
        """Find for each node in the graph what its associated layer in `model` is.

        Parameters
        ----------
        - model (nn.Module): the `spiking_model` used as argument to the class instance.

        Returns
        ----------
        - indx_2_module_map (dict): the mapping between a node (`key` as an `int`) and its module (`value` as a `nn.Module`).
        """

        indx_2_module_map = dict()

        for name, module in model.named_modules():
            # Make sure names match those provided by nirtorch nodes
            if name in self._name_2_indx_map:
                indx_2_module_map[self._name_2_indx_map[name]] = module
            else:
                # In older nirtorch versions, node names are "sanitized"
                # Try with sanitized version of the name
                name = nirtorch.utils.sanitize_name(name)
                if name in self._name_2_indx_map:
                    indx_2_module_map[self._name_2_indx_map[name]] = module

        return indx_2_module_map

    def _update_internal_representation(self, remapped_nodes: Dict[int, int]):
        """Update internal attributes after remapping of nodes

        Parameters
        ----------
        remapped_nodes (dict): Maps previous (key) to new (value) node
            indices. Nodes that were removed are not included.
        """

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
        self._indx_2_module_map = {
            remapped_nodes[old_idx]: module
            for old_idx, module in self._indx_2_module_map.items()
            if old_idx in remapped_nodes
        }

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

            if isinstance(self.indx_2_module_map[node], sinabs.layers.merge.Merge):
                # find `Merge` arguments (at this point the inputs to Merge should have been calculated).
                input_nodes = self._find_merge_arguments(node)

                # retrieve arguments output tensors.
                inputs = [nodes_io_map[n]["output"] for n in input_nodes]

                # TODO - this is currently a limitation imposed by the validation checks done by Speck once a configuration: it wants
                # different input sources to a core to have the same output shapes.
                if any(inp.shape != inputs[0].shape for inp in inputs):
                    raise ValueError(
                        f"Layer `sinabs.layers.merge.Merge` (node {node}) requires input tensors with the same shape"
                    )

                # forward input through the node.
                _output = self.indx_2_module_map[node](*inputs)

                # save node's I/O tensors.
                nodes_io_map[node] = {"input": inputs[0], "output": _output}

            else:

                if node in self._entry_nodes:
                    # forward input dummy through node.
                    _output = self.indx_2_module_map[node](input_dummy)

                    # save node's I/O tensors.
                    nodes_io_map[node] = {"input": input_dummy, "output": _output}

                else:
                    # find node generating the input to be used.
                    input_node = self._find_source_of_input_to(node)
                    _input = nodes_io_map[input_node]["output"]

                    # forward input through the node.
                    _output = self.indx_2_module_map[node](_input)

                    # save node's I/O tensors.
                    nodes_io_map[node] = {"input": _input, "output": _output}

        # replace the I/O tensor information by its shape information, ignoring the batch/time axis
        for node, io in nodes_io_map.items():
            input_shape = io["input"].shape[1:]
            output_shape = io["output"].shape[1:]
            # Linear layers have fewer in/out dimensions. Extend by appending 1's
            if (length := len(input_shape)) < 3:
                input_shape = (*input_shape, *(1 for __ in range(3 - length)))
            assert len(input_shape) == 3
            if (length := len(output_shape)) < 3:
                output_shape = (*output_shape, *(1 for __ in range(3 - length)))
            assert len(output_shape) == 3
            nodes_io_map[node]["input"] = input_shape
            nodes_io_map[node]["output"] = output_shape

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

    def _find_merge_arguments(self, node: int) -> Edge:
        """A `Merge` layer receives two inputs. Return the two inputs to `merge_node` representing a `Merge` layer.

        Returns
        ----------
        - args (tuple): the IDs of the nodes that provice the input arguments to a `Merge` layer.
        """
        sources = self._find_all_sources_of_input_to(node)

        if len(sources) != 2:
            raise ValueError(
                f"Number of arguments found for `Merge` node {node} is {len(sources)} (should be 2)."
            )

        return tuple(sources)

    def _find_valid_targets(
        self, node: int, ignored_node_classes: Tuple[Type] = ()
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
                if isinstance(self.indx_2_module_map[tgt], ignored_node_classes):
                    # Find valid targets of target
                    targets.update(self._find_valid_targets(tgt, ignored_node_classes))
                else:
                    # Target is valid, add it to `targets`
                    targets.add(tgt)
        return targets
