from copy import deepcopy
from typing import Dict, Optional, Tuple, Union
from warnings import warn

import numpy as np
import torch
from torch import nn

import sinabs.activation
import sinabs.layers as sl

from .discretize import discretize_conv_spike_
from .dvs_layer import expand_to_pair

class DynapcnnLayer(nn.Module):
    """Create a DynapcnnLayer object representing a dynapcnn layer. """

    def __init__(
        self,
        dpcnnl_index: int,
        dcnnl_data: dict, 
        discretize: bool,
        sinabs_edges: list
    ):
        super().__init__()
        """
        ...

        Parameters
        ----------
            dpcnnl_index (int): ...
            dcnnl_data (dict): ...
            discretize (bool): ...
            sinabs_edges (list): ...
        """
        self.dpcnnl_index = dpcnnl_index
        self.assigned_core = None

        if 'core_idx' in dcnnl_data:
            self.assigned_core = dcnnl_data['core_idx']

        self.lin_to_conv_conversion = False

        conv = None
        self.conv_node_id = None
        self.conv_in_shape = None
        self.conv_out_shape = None

        spk = None
        self.spk_node_id = None

        pool = []
        self.pool_node_id = []
        
        self.dynapcnnlayer_destination = dcnnl_data['destinations']

        for key, value in dcnnl_data.items():
            if isinstance(key, int):
                # value has data pertaining a node (torch/sinabs layer).
                if isinstance(value['layer'], sl.IAFSqueeze):
                    spk = value['layer']
                    self.spk_node_id = key
                elif isinstance(value['layer'], nn.Linear) or isinstance(value['layer'], nn.Conv2d):
                    conv = value['layer']
                    self.conv_node_id = key
                elif isinstance(value['layer'], sl.SumPool2d):
                    pool.append(value['layer'])
                    self.pool_node_id.append(key)
                else:
                    raise ValueError(f'Node {key} has not valid layer associated with it.')
                
        if not conv:
            raise ValueError(f'Convolution layer not present.')
        
        if not spk:
            raise ValueError(f'Spiking layer not present.')
        
        spk = deepcopy(spk)
        if spk.is_state_initialised():
            # TODO this line bellow is causing an exception on `.v_men.shape` to be raised in `.get_layer_config_dict()`. Find out why.
            # spk.v_mem = spk.v_mem.data.unsqueeze(-1).unsqueeze(-1)      # expand dims.

            # TODO hacky stuff: make it better (THIS SEEMS TO BE FIXING THE PROBLEM ABOVE THO).
            if len(list(spk.v_mem.shape)) != 4:
                spk.v_mem = spk.v_mem.data.unsqueeze(-1).unsqueeze(-1)      # expand dims.

        if isinstance(conv, nn.Linear):
            conv, conv_in_shape = self._convert_linear_to_conv(conv, dcnnl_data[self.conv_node_id])

            # the original `nn.Linear` output shape becomes the equivalent `nn.Conv2d` shape.
            self.conv_out_shape = self._update_conv_node_output_shape(
                conv_layer=conv, layer_data=dcnnl_data[self.conv_node_id], input_shape=conv_in_shape)

            # the I/O shapes for neuron layer following the new conv need also to be updated.
            self._update_neuron_node_output_shape(layer_data=dcnnl_data[self.spk_node_id], input_shape=self.conv_out_shape)

        else:
            self.conv_out_shape = dcnnl_data[self.conv_node_id]['output_shape']
            conv = deepcopy(conv)

        # check if convolution kernel is a square.
        if conv.kernel_size[0] != conv.kernel_size[1]:
            raise ValueError('The kernel of a `nn.Conv2d` must have the same height and width.')

        # input shape of conv layer.
        self.conv_in_shape = dcnnl_data[self.conv_node_id]['input_shape']

        # this weight rescale comes from the node projecting into this 'conv' node.
        if dcnnl_data['conv_rescale_factor'] != 1:
            # this has to be done after copying but before discretizing
            conv.weight.data = (conv.weight / dcnnl_data['conv_rescale_factor']).clone().detach()

        # int conversion is done while writing the config.
        if discretize:
            conv, spk = discretize_conv_spike_(conv, spk, to_int=False)

        # consolidate layers.
        self.conv_layer = conv
        self.spk_layer = spk
        self.pool_layer = []

        if len(pool) != 0:
            # the 1st pooling targets the 1st destination in `dcnnl_data['destinations']`, the 2nd pooling targets the 2nd destination...
            for plyr in pool:
                # @TODO POSSIBLE INCONSISTENCY: if the `SumPool2d` is the result of a conversion from `AvgPool2d` then `SumPool2d.kernel_size` 
                # is of type tuple, otherwise it is an int.
                if isinstance(plyr.kernel_size, tuple) and plyr.kernel_size[0] != plyr.kernel_size[1]:
                    raise ValueError("Only square kernels are supported")
                self.pool_layer.append(deepcopy(plyr))

        # map destination nodes for each layer in this instance.
        self.nodes_destinations = self._get_destinations_input_source(sinabs_edges)

    ####################################################### Public Methods #######################################################

    def get_destination_dcnnl_index(self, dcnnl_id: int) -> int:
        """ The `forward` method will return as many tensors as there are elements in `self.dynapcnnlayer_destination`. Since the i-th returned tensor is to be
        fed to the i-th destionation in `self.dynapcnnlayer_destination`, the return of this method can be used to index a tensor returned in the `forward` method.

        Parameters
        ----------
            dcnnl_id (int): this should be one of the values listed within `self.dynapcnnlayer_destination`.

        Returns
        ----------
            The index of `dcnnl_id` within `self.dynapcnnlayer_destination`.
        """
        return self.dynapcnnlayer_destination.index(dcnnl_id)
    
    def forward(self, x):
        """Torch forward pass.

        Returns
        ----------
            This method will return as many tensors as there are destinations associated with this instance. The returned tensors always follows the 
            sequence `return spiking_layer_output, pooling_layer_1, pooling_layer_2, ...`.
        
        Example
        ----------
            With `self.nodes_destinations = {1: [5, 8], 2: [4], 3: [7]}` this method will return 4 tensors (each to be sent to a different DynapcnnLayer): the 1st 
            and 2nd are the outputs of the spiking layer (`node 1`); the 3rd is the output of the first pooling layer (`node 2`, receiving the output of node 1) appearing 
            right after the spiking layer as defined in the original `nn.Module` being converted to a `DynapcnnNetwork`; the 4th is the output of the second pooling 
            layer (`node 3`) to appear after the spiking layer. Thus, the return will be `return node_1_out, node_1_out, node_2_out, node_3_out`. This means the edges 
            in the computational graph involved in this mapping were:
            
            1 --> 2     # `2` is one of the pooling layers of this DynapcnnLayer.
            1 --> 3     # `3` is one of the pooling layers of this DynapcnnLayer.
            1 --> 5     # `5` is a conv layer belonging to another DynapcnnLayer U.
            1 --> 8     # `8` is a conv layer belonging to another DynapcnnLayer V.
            2 --> 4     # `4` is a conv layer belonging to another DynapcnnLayer X.
            3 --> 7     # `7` is a conv layer belonging to another DynapcnnLayer Y.
        """

        returns = []
        
        x = self.conv_layer(x)
        x = self.spk_layer(x)

        # pooling layers are sequentially added to `self.pool_layer` so we'll pass data to them sequentially.
        pooling_indexer = 0

        # building return set of all layers as they appear in `self.nodes_destinations`.
        for node_id, destination_node_list in self.nodes_destinations.items():
            if node_id == self.spk_node_id:
                # spiking output for each node outside this DynapcnnLayer receiving from its spiking layer.
                for _ in destination_node_list:
                    returns.append(x)
            else:
                # returns of each pooling layer are arranged sequenatially.
                for _ in destination_node_list:
                    ith_pool_output = self.pool_layer[pooling_indexer](x)
                    returns.append(ith_pool_output)
                
                # forward through next pooling layer in `self.pool_layer` in the next iteration.
                pooling_indexer += 1

        if len(returns) != len(self.dynapcnnlayer_destination):
            raise ValueError(f'Number of returned tensors ({len(returns)}) differ from the number of destinations ({len(self.dynapcnnlayer_destination)}).')
        
        if len(returns) == 0 and len(self.pool_layer) == 0:
            # this is the output layer and there's no pooling after the neurons.
            returns.append(x)
        elif len(returns) == 0 and len(self.pool_layer) == 1:
            # this is the output layer and there's 1 pooling after the neurons.
            returns.append(self.pool_layer[0](x))
        elif len(returns) == 0 and len(self.pool_layer) > 1:
            raise ValueError(f'Output DynapcnnLayer starting with node {self.conv_node_id} has {len(self.pool_layer)} pooling layers: it should have either 1 or none.')
        else:
            pass

        return tuple(returns)

    def get_modified_node_it(self, dcnnl_data: dict) -> Union[Tuple[int, tuple], Tuple[None, None]]:
        """ ."""
        if self.lin_to_conv_conversion:
            return self.spk_node_id, dcnnl_data[self.spk_node_id]['output_shape']
        return None, None
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        return self.spk_layer.zero_grad(set_to_none)
    
    def get_conv_output_shape(self, conv_layer: nn.Conv2d, input_shape: tuple):
        """ ."""
        # get the layer's parameters.
        out_channels = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size
        stride = conv_layer.stride
        padding = conv_layer.padding
        dilation = conv_layer.dilation

        # compute the output height and width.
        out_height = ((input_shape[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
        out_width = ((input_shape[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1

        return (out_channels, out_height, out_width)

    def summary(self) -> dict:
        # TODO I can't see pooling being used in checking memory constraints by the builder so I'm ignoring for now the fact that multiple pooling could exist.
        
        _pool = None

        # @TODO POSSIBLE INCONSISTENCY: if the `SumPool2d` is the result of a conversion from `AvgPool2d` then `SumPool2d.kernel_size` 
        # is of type tuple, otherwise it is an int. 
        if len(self.pool_layer) != 0:
            # @TODO ignoring for now that there could be multiple poolings (just use the first one).
            if isinstance(self.pool_layer[0].kernel_size, tuple):
                _pool = list(self.pool_layer[0].kernel_size)
            elif isinstance(self.pool_layer[0].kernel_size, int):
                _pool = [self.pool_layer[0].kernel_size, self.pool_layer[0].kernel_size]
            else:
                raise ValueError('Type of `self.pool_layer[0].kernel_size` not understood.')

        return {
            "pool": (_pool),
            "kernel": list(self.conv_layer.weight.data.shape),
            "neuron": self.conv_out_shape,                          # neuron layer output has the same shape as the convolution layer ouput.
        }

    def get_layer_config_dict(self) -> dict:
        """ Returns a dict containing the properties required to configure a `CNNLayerConfig` instance that
        will map this DynapcnnLayer onto the chip.
        """
        config_dict = {}

        # configures `CNNLayerConfig.dimensions` (instance of `CNNLayerDimensions`).
        dimensions = {}

        # input shape of convolution.
        dimensions['input_shape'] = {
            'size': {'x': self.conv_in_shape[2], 'y': self.conv_in_shape[1]},
            'feature_count': self.conv_in_shape[0]
            }
        
        # ouput shape of convolution.
        dimensions['output_shape'] = {
            'size': {'x': self.conv_out_shape[2], 'y': self.conv_out_shape[1]},
            'feature_count': self.conv_out_shape[0]
            }
        
        # convolution padding, stride and kernel sizes.
        dimensions['padding']       = {'x': self.conv_layer.padding[1], 'y': self.conv_layer.padding[0]}
        dimensions['stride']        = {'x': self.conv_layer.stride[1], 'y': self.conv_layer.stride[0]}
        dimensions['kernel_size']   = self.conv_layer.kernel_size[0]

        config_dict['dimensions'] = dimensions              # update config dict.

        # update parameters from convolution.
        if self.conv_layer.bias is not None:
            (weights, biases) = self.conv_layer.parameters()
        else:
            (weights,) = self.conv_layer.parameters()
            biases = torch.zeros(self.conv_layer.out_channels)

        # parameters of the convolution in the DynapcnnLayer.

        weights = weights.transpose(2, 3)                   # need this to match samna convention.
        config_dict['weights'] = weights.int().tolist()     # 4-D list of lists representing kernel parameters.
        config_dict['biases'] = biases.int().tolist()
        config_dict['leak_enable'] = biases.bool().any()

        # parameters of the neurons in the DynapcnnLayer.

        # set neuron states.                                # TODO coppied from the old implementation.
        if not self.spk_layer.is_state_initialised():
            # then we assign no initial neuron state to DYNAP-CNN.
            f, h, w = self.conv_out_shape                   # same as the convolution layer.
            neurons_state = torch.zeros(f, w, h)

        elif self.spk_layer.v_mem.dim() == 4:
            # 4-D states should be the norm when there is a batch dim.
            neurons_state = self.spk_layer.v_mem.transpose(2, 3)[0]

        else:
            raise ValueError(f"Current v_mem (shape: {self.spk_layer.v_mem.shape}) of spiking layer not understood.")
            # TODO error here: find where `self.spk_layer.v_mem` is being initialized.
        
        # resetting vs returning to 0.                       # TODO coppied from the old implementation.
        if isinstance(self.spk_layer.reset_fn, sinabs.activation.MembraneReset):
            return_to_zero = True                            # neurons in this layer will return to 0 when firing.
        elif isinstance(self.spk_layer.reset_fn, sinabs.activation.MembraneSubtract):
            return_to_zero = False                           # threshold will be subtracted from the value their membrane potential reached before firing.
        else:
            raise Exception("Unknown reset mechanism. Only MembraneReset and MembraneSubtract are currently understood.")
        
        if self.spk_layer.min_v_mem is None:
            min_v_mem = -(2**15)
        else:
            min_v_mem = int(self.spk_layer.min_v_mem)

        # set neuron configuration for this DynapcnnLayer.
        config_dict.update(
            {
                "return_to_zero": return_to_zero,
                "threshold_high": int(self.spk_layer.spike_threshold),
                "threshold_low": min_v_mem,
                "monitor_enable": False,
                "neurons_initial_value": neurons_state.int().tolist()
            }
        )

        # set pooling configuration for each destinaition. This configures a `CNNLayerConfig.destinations` (instance of `CNNLayerDimensions`).
        config_dict['destinations'] = []
        if len(self.pool_layer) != 0:
            for i in range(len(self.pool_layer)):
                dest_config = {
                    'layer': self.dynapcnnlayer_destination[i],# TODO this destination index is not the core index yet, just the index of the DynapcnnLayers themselves.
                    'enable': True, 
                    'pooling': self.pool_layer[i].kernel_size[0] if isinstance(self.pool_layer[i].kernel_size, tuple) else self.pool_layer[i].kernel_size    # TODO make sure the kernel is a square.
                    }
                
                config_dict['destinations'].append(dest_config)

        # setting of the kill bits need to be done outside this method.

        return config_dict
    
    def memory_summary(self):
        """Computes the amount of memory required for each of the components. Note that this is not
        necessarily the same as the number of parameters due to some architecture design
        constraints.

        .. math::

            K_{MT} = c \\cdot 2^{\\lceil \\log_2\\left(k_xk_y\\right) \\rceil + \\lceil \\log_2\\left(f\\right) \\rceil}

        .. math::

            N_{MT} = f \\cdot 2^{ \\lceil \\log_2\\left(f_y\\right) \\rceil + \\lceil \\log_2\\left(f_x\\right) \\rceil }

        Returns
        -------
        A dictionary with keys kernel, neuron and bias and the corresponding memory sizes
        """
        summary = self.summary()
        f, c, h, w = summary["kernel"]
        f, neuron_height, neuron_width = self.conv_out_shape        # neuron layer output has the same shape as the convolution layer ouput.

        return {
            "kernel": c * pow(2, np.ceil(np.log2(h * w)) + np.ceil(np.log2(f))),
            "neuron": f
            * pow(2, np.ceil(np.log2(neuron_height)) + np.ceil(np.log2(neuron_width))),
            "bias": 0 if self.conv_layer.bias is None else len(self.conv_layer.bias),
        }
    
    def __str__(self):
        pretty_print = '\n'

        pretty_print += 'COMPUTATIONAL NODES:\n\n'

        pretty_print += f'(node {self.conv_node_id}): {self.conv_layer}\n'
        pretty_print += f'(node {self.spk_node_id}): {self.spk_layer}'
        if len(self.pool_layer) != 0:
            for idx, lyr in enumerate(self.pool_layer):
                pretty_print += f'\n(node {self.pool_node_id[idx]}): {lyr}'

        pretty_print += '\n\nMETADATA:\n'
        pretty_print += f'\n> assigned core index: {self.assigned_core}'
        pretty_print += f'\n> destination DynapcnnLayers: {self.dynapcnnlayer_destination}'

        for node, destinations in self.nodes_destinations.items():
            pretty_print += f'\n> node {node} feeds input to nodes {destinations}'

        return pretty_print
    
    ####################################################### Private Methods #######################################################

    def _update_neuron_node_output_shape(self, layer_data: dict, input_shape: tuple) -> None:
        """ Following the conversion of a `nn.Linear` into a `nn.Conv2d` the neuron layer in the
        sequence also needs its I/O shapes uodated.
        """
        layer_data['input_shape'] = input_shape
        layer_data['output_shape'] = layer_data['input_shape']

    def _update_conv_node_output_shape(self, conv_layer: nn.Conv2d, layer_data: dict, input_shape: tuple) -> Tuple:
        """ The input shapes to nodes are extracted using a list of edges by finding the output shape of the 1st element
        in the edge and setting it as the input shape to the 2nd element in the edge. If a node used to be a `nn.Linear` 
        and it became a `nn.Conv2d`, output shape in the mapper needs to be updated, otherwise there will be a mismatch
        between its output and the input it provides to another node.
        """
        layer_data['output_shape'] = self.get_conv_output_shape(conv_layer, input_shape)

        return layer_data['output_shape']
    
    def _convert_linear_to_conv(self, lin: nn.Linear, layer_data: dict) -> nn.Conv2d:
        """Convert Linear layer to Conv2d.

        Parameters
        ----------
            lin: nn.Linear
                Linear layer to be converted

        Returns
        -------
            nn.Conv2d
                Convolutional layer equivalent to `lin`.
        """
        self.lin_to_conv_conversion = True

        input_shape = layer_data['input_shape']

        in_chan, in_h, in_w = input_shape

        if lin.in_features != in_chan * in_h * in_w:
            raise ValueError("Shapes don't match.")

        layer = nn.Conv2d(
            in_channels=in_chan,
            kernel_size=(in_h, in_w),
            out_channels=lin.out_features,
            padding=0,
            bias=lin.bias is not None,
        )

        if lin.bias is not None:
            layer.bias.data = lin.bias.data.clone().detach()

        layer.weight.data = (
            lin.weight.data.clone()
            .detach()
            .reshape((lin.out_features, in_chan, in_h, in_w))
        )

        return layer, input_shape
    
    def _get_destinations_input_source(self, sinabs_edges: list) -> dict:
        """ Creates a mapping between each layer in this `DynapcnnLayer` instance and its targe nodes that are part of different
        `DynapcnnLayer` instances. This mapping is used to figure out how many tensors the `forward` method needs to return.
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