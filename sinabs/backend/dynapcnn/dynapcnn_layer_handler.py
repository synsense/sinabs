# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

from typing import Dict, List, Tuple, Union

from torch import nn


class DynapcnnLayerHandler:
    """
    Class handling the pre-processing of network-level data into (device) layer-level data (i.e., arguments required for a `DynapcnnLayer` instantiation).

    Parameters
    ----------
    - dpcnnl_index (int): the index/ID that will be associated with a `DynapcnnLayer` instance. This integer indexes a `dict` within `dcnnl_data`
        that comprises a set of layers (`nn.Module`) and their respective I/O tensor shapes.
    - dcnnl_data (dict): contains the nodes to be merged into this `DynapcnnLayer`, their I/O shapes and the index of the other `DynapcnnLayer`s to
        be set as destinations. The `int` keys correspond to the nodes IDs associated `nn.Module`s (a single layer in the original network) becoming
        part of this `DynapcnnLayer` instance, while the `str` keys correspond to this instance's destinations and re-scaling factors.
    - discretize (bool): whether or not the weights/neuron parameters of the model will be quantized.
    - sinabs_edges (list): each `nn.Module` within `dcnnl_data` is a node in the original computational graph describing a spiking network. An edge
        `(A, B)` describes how modules forward data amongst themselves. This list is used by a `DynapcnnLayer` to figure out the number and
        sequence of output tesnors its forward method needs to return.
    - weight_rescaling_fn (callable): a method that handles how the re-scaling factor for one or more `SumPool2d` projecting to
        the same convolutional layer are combined/re-scaled before applying them.
    - entry_nodes (list): node IDs corresponding to layers in the original network that are input nodes (i.e., a "point of entry" for the external data).
    """

    def __init__(
        self,
        dpcnnl_index: int,
        dcnnl_data: Dict[
            Union[int, str],
            Union[
                Dict[str, Union[nn.Module, Tuple[int, int, int], Tuple[int, int, int]]],
                List[int],
            ],
        ],
        sinabs_edges: List[Tuple[int, int]],
        entry_nodes: List[int],
    ):
        self.dpcnnl_index = dpcnnl_index
        self.entry_point = False

        if "core_idx" in dcnnl_data:
            self.assigned_core = dcnnl_data["core_idx"]
        else:
            self.assigned_core = None

        self.dynapcnnlayer_destination = dcnnl_data["destinations"]

        # map destination nodes for each layer in this instance.
        self.nodes_destinations = self._get_destinations_input_source(sinabs_edges)

        # TODO: Move detection of entry points to dcnnl_info generation
        # flag if the instance is an entry point (i.e., an input node of the network).
        if self.conv_node_id in entry_nodes:
            self.entry_point = True

    ####################################################### Public Methods #######################################################

    def get_destination_dcnnl_index(self, dcnnl_id: int) -> int:
        """The `forward` method will return as many tensors as there are elements in `self.dynapcnnlayer_destination`. Since the i-th returned tensor is to be
        fed to the i-th destionation in `self.dynapcnnlayer_destination`, the return of this method can be used to index a tensor returned in the `forward` method.

        Parameters
        ----------
        - dcnnl_id (int): this should be one of the values listed within `self.dynapcnnlayer_destination`.

        Returns
        ----------
        - The index of `dcnnl_id` within `self.dynapcnnlayer_destination`.
        """
        return self.dynapcnnlayer_destination.index(dcnnl_id)

    def __str__(self):
        pretty_print = "\n"

        pretty_print += "COMPUTATIONAL NODES:\n\n"

        pretty_print += f"(node {self.conv_node_id}): {self.conv_layer}\n"
        pretty_print += f"(node {self.spk_node_id}): {self.spk_layer}"
        if len(self.pool_layer) != 0:
            for idx, lyr in enumerate(self.pool_layer):
                pretty_print += f"\n(node {self.pool_node_id[idx]}): {lyr}"

        pretty_print += "\n\nMETADATA:\n"
        pretty_print += f"\n> network's entry point: {self.entry_point}"
        pretty_print += (
            f"\n> convolution's weight re-scaling factor: {self.conv_rescaling_factor}"
        )
        pretty_print += f"\n> assigned core index: {self.assigned_core}"
        pretty_print += (
            f"\n> destination DynapcnnLayers: {self.dynapcnnlayer_destination}"
        )

        for node, destinations in self.nodes_destinations.items():
            pretty_print += f"\n> node {node} feeds input to nodes {destinations}"

        return pretty_print

    ####################################################### Private Methods #######################################################

    def _get_destinations_input_source(self, sinabs_edges: list) -> dict:
        """Creates a mapping between each layer in this `DynapcnnLayer` instance and its targe nodes that are part of different
        `DynapcnnLayer` instances. This mapping is used to figure out how many tensors the `forward` method needs to return.

        Parameters
        ----------
        - sinabs_edges (list): each `nn.Module` within `dcnnl_data` is a node in the original computational graph describing a spiking
            network. An edge `(A, B)` describes how modules forward data amongst themselves. This list is used by a `DynapcnnLayer` to
            figure out the number and sequence of output tesnors its forward method needs to return.

        Returns
        ----------
        - destinations_input_source (dict): maps a `nn.Module` within this `DynapcnnLayer` to the nodes it provides the input to.
        """
        destinations_input_source = {}

        # check whether spiking layer projects outside this DynapcnnLayer (i.e. to one of the destinations without passing through a pooling).
        spk_destinations = []
        for edge in sinabs_edges:
            if edge[0] == self.spk_node_id and edge[1] not in self.pool_node_id:
                # spiking layer projects to a node outside this DynapcnnLayer.
                spk_destinations.append(edge[1])
        if len(spk_destinations) > 0:
            destinations_input_source[self.spk_node_id] = []
            for node_id in spk_destinations:
                destinations_input_source[self.spk_node_id].append(node_id)

        #   get `pooling->destination` mapping. The pooling outputs will be arranged sequentially since the pooling layers are added sequentially
        # to `self.pool_layer` (i.e., as they appear in the computational graph of the original `nn.Module`).
        for id in self.pool_node_id:
            destinations_input_source[id] = []
            for edge in sinabs_edges:
                if edge[0] == id:
                    destinations_input_source[id].append(edge[1])

        return destinations_input_source

    @staticmethod
    def find_nodes_core_id(node: int, all_handlers: dict) -> int:
        """Loops through all handlers in `all_handlers` to find to which core a `DynapcnnLayer` containing
        `node` has been assigned to."""

        for _, dcnnl in all_handlers.items():

            if (
                node == dcnnl["layer_handler"].conv_node_id
                or node == dcnnl["layer_handler"].spk_node_id
                or node in dcnnl["layer_handler"].pool_node_id
            ):
                return dcnnl["layer_handler"].assigned_core

        raise ValueError(f"Node {node} not found in any of the cores.")
