import samna
import numpy as np
import torch.nn as nn
import sinabs.layers as sl
import time
from typing import List, Tuple, Union

from samna.specksim.nodes import SpecksimConvolutionalFilterNode as ConvFilter
from samna.specksim.nodes import SpecksimIAFFilterNode as IAFFilter
from samna.specksim.nodes import SpecksimSumPoolingFilterNode as SumPoolFilter

to_tuple = lambda x: (x, x) if isinstance(x, int) else x  

def convert_linear_to_convolutional(layer: nn.Linear, input_shape: Tuple[int, int, int]) -> nn.Conv2d:
    """Converts a linear layer to a convolutional layer.
    For original refer to: sinabs/backend/dynapcnn/dynapcnn_layer.py::DynapcnnLayer::_convert_linear_to_conv

    Args:
        layer (nn.Linear): A linear layer 
        input_shape (Tuple[int, int, int]): Input shape in (channel, y, x)

    Returns:
        nn.Conv2d: A convolutional layer 
    """
    in_channels, in_height, in_width = input_shape

    if layer.in_features != in_channels * in_height * in_width:
        raise ValueError(f"Linear layer has {layer.in_features} features." +
                         f"However, the input to network has {in_channels * in_height * in_width}")
    
    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        kernel_size=(in_height, in_width),
        out_channels=layer.out_features,
        padding=0,
        bias=False
    )
    conv_layer.weight.data = (
        layer.weight.data.clone().detach().reshape(
            (layer.out_features, in_channels, in_height, in_width)
        )
    )
    return conv_layer


def convert_convolutional_layer(
    layer: nn.Conv2d,
    input_shape: Tuple[int, int, int]
) -> Tuple[ConvFilter, Tuple[int, int, int]]:
    """Convert a convolutional layer to samna filter

    Args:
        layer (nn.Conv2d): A PyTorch Convolutional Layer. The biases has to be disabled. 
        input_shape (Tuple[int, int, int]): Input shape of the layer in (channel, y, x)

    Returns:
        Tuple[ConvFilter, Tuple[int, int, int]]: Returns the samna filter and the output shape
        of the layer 
    """
    if layer.bias:
        raise ValueError("Biases are not supported!")

    # Create the filter node
    conv_filter = ConvFilter()
    
    # Extract the layer parameters 
    in_channels = int(layer.in_channels)
    out_channels = int(layer.out_channels)
    in_shape = (input_shape[1], input_shape[2]) 
    kernel_size = to_tuple(layer.kernel_size) 
    stride = to_tuple(layer.stride) 
    padding = to_tuple(layer.padding) 
    
    # Set the filter parameters
    conv_filter.set_parameters(
        in_channels,
        out_channels,
        kernel_size,
        in_shape,
        stride,
        padding
    ) 
    
    # Set the layer weights
    conv_filter.set_weights(layer.weight.cpu().detach().tolist())

    # Get the output image shape
    output_shape = [out_channels, *conv_filter.get_layer().get_output_shape()]
    
    return conv_filter, output_shape

def convert_pooling_layer(
    layer: Union[nn.AvgPool2d, sl.SumPool2d],
    input_shape: Tuple[int, int, int]
) -> Tuple[SumPoolFilter, Tuple[int, int, int]]:

    if to_tuple(layer.kernel_size) != to_tuple(layer.stride):
        raise ValueError("For pooling layers kernel size has to be the same as the stride")
    
    # initialize filter
    pooling_filter = SumPoolFilter()

    # extract parameters from layer
    kernel_size = to_tuple(layer.kernel_size)
    in_shape = tuple(input_shape[1:])

    # set the filter parameters
    pooling_filter.set_parameters(
        kernel_size,
        in_shape
    )
    
    # get the output shape
    output_shape = [input_shape[0], *pooling_filter.get_layer().get_output_shape()]

    return pooling_filter, output_shape 

def convert_iaf_layer(
    layer: Union[sl.IAF, sl.IAFSqueeze],
    input_shape: Tuple[int, int, int]
) -> Tuple[IAFFilter, Tuple[int, int, int]]:
    """Convert a sinabs IAF layer into a specksim IAF Filter
    
    Args:
        layer (Union[sl.IAF, sl.IAFSqueeze]): A Sinabs IAF layer 
        input_shape (Tuple[int, int, int]): Input shape in (channel, y, x) 

    Returns:
        Tuple[IAFFilter, Tuple[int, int, int]]: A specksim IAF Filter and the output shape 
    """
    iaf_filter = IAFFilter()
    
    n_channels = input_shape[0]
    in_shape = tuple(input_shape[1:])
    spike_threshold = float(layer.spike_threshold)
    min_v_mem = float(layer.min_v_mem)
    iaf_filter.set_parameters(
        n_channels,
        in_shape,
        spike_threshold,
        min_v_mem 
    )
    return iaf_filter, input_shape


def from_sinabs(
    network: nn.Sequential, 
    input_shape: Tuple[int, int, int]
) -> "SpecksimNetwork":
    """Convert a sinabs network to a SpecksimNetwork

    Args:
        network (nn.Sequential):  A sequential sinabs model. 
        input_shape (Tuple[int, int, int]): Network input shape in channel, y, x 

    Returns:
        SpecksimNetwork: A container for the samna event-based filter that simualtes
            the network 
    """
    graph = samna.graph.EventFilterGraph()
    filters = []
    current_shape = list(input_shape)

    # Add an input node
    filters.append(samna.BasicSourceNode_specksim_events_spike())
    
    for layer in network:
        if isinstance(layer, nn.Conv2d):
            samna_filter, current_shape = convert_convolutional_layer(layer, current_shape)
        elif isinstance(layer, (sl.SumPool2d, nn.AvgPool2d)):
            samna_filter, current_shape = convert_pooling_layer(layer, current_shape)
        elif isinstance(layer, (sl.IAF, sl.IAFSqueeze)):
            samna_filter, current_shape = convert_iaf_layer(layer, current_shape) 
        elif isinstance(layer, nn.Linear):
            conv_layer = convert_linear_to_convolutional(layer, current_shape)
            samna_filter, current_shape = convert_convolutional_layer(conv_layer, current_shape)
        elif isinstance(layer, nn.Flatten):
            continue 
        else:
            raise TypeError(f"Only Conv2d, SumPool2d and IAF layers are supported: {layer}")

        filters.append(samna_filter)
    
    # Add an output node
    filters.append(samna.BasicSinkNode_specksim_events_spike())
    members = graph.sequential(filters)
    
    return SpecksimNetwork(graph, members) 

class SpecksimNetwork:
    def __init__(
        self,
        graph: samna.graph.EventFilterGraph, 
        graph_members: List["SamnaFilterNode"],
        sleep_duration: float = 0.1
    ):
        """Specksim simulation container object.

        Args:
            graph (samna.graph.EventFilterGraph): A samna graph that contains the network layers as samna filters.
            graph_members (List["SamnaFilterNode"]): A list of samna filters.
            sleep_duration (float): Duration between each read from the graph in seconds. Defaults to 0.1.
        """
        self.network: samna.graph.EventFilterGraph = graph 
        self.members = graph_members
        self.sleep_duration = sleep_duration
        self.xytp_dtype = None
    
    
    def forward(self, xytp: np.record) -> np.record:
        """Applies the network forward pass given events. 

        Args:
            xytp (np.record): Input events as a numpy record array with keys ("x", "y", "t", "p") 

        Returns:
            np.record: Output events as a numpy record array of the same type as the input events. 
        """
        # set up container for output spikes
        output_spikes = []
        # convert xytp to specksim spikes
        spikes = self.xytp_to_specksim_spikes(xytp)
        # start the network graph 
        self.network.start()
        # do the forward pass
        self.members[0].write(spikes)

        # continuously read from the stream until there are no new
        # events left in the stream
        while True:
            time.sleep(self.sleep_duration)
            previous_length = len(output_spikes)
            output_spikes.extend(self.members[-1].get_events())
            current_length = len(output_spikes)
            if current_length == previous_length:
                break
        # stop the streaming graph at the end 
        self.network.stop()
        return self.specksim_spikes_to_xytp(output_spikes, xytp.dtype)
    
    def __call__(self, xytp: np.record) -> np.record:
        return self.forward(xytp)
    
    @staticmethod
    def xytp_to_specksim_spikes(xytp: np.record) -> List[samna.specksim.events.Spike]:
        """Takes in xytp and returns a list of spikes compatible with specksim

        Args:
            xytp (np.record): A numpy record array with 4 keys 'x', 'y', 't' and 'p'

        Returns:
            List[samna.specksim.events.Spike]: A list of specksim compatible spike events 
        """
        spikes = []
        for event in xytp:
            x, y, t, p = event["x"], event["y"], event["t"], event["p"]
            spike = samna.specksim.events.Spike(p, y, x, t)
            spikes.append(spike) 
        return spikes
    
    @staticmethod
    def specksim_spikes_to_xytp(spikes: List[samna.specksim.events.Spike], output_dtype: np.dtype) -> np.record:
        """Takes in specksim spikes and converts them to xytp of the same type as input

        Args:
            spikes (List[samna.specksim.events.Spike]): A list of specksim spikes coming from the output
            of the network. 
            output_dtype (np.dtype): A numpy dtype with keys ("x", "y", "t", "p") in this order.

        Returns:
            np.record: A record array of the given output type 
        """
        output_events = []
        for spike in spikes:
            x, y, t, p = spike.x, spike.y, spike.timestamp, spike.feature
            output_event = np.array([x, y, t, p], dtype=output_dtype)
            output_events.append(output_event)
        return np.array(output_events, dtype=output_dtype)