import time
from typing import Dict, List, Tuple, Union
from warnings import warn

import numpy as np
import samna
import torch.nn as nn
from samna.specksim.nodes import SpecksimConvolutionalFilterNode as ConvFilter
from samna.specksim.nodes import SpecksimIAFFilterNode as IAFFilter
from samna.specksim.nodes import SpecksimSumPoolingFilterNode as SumPoolFilter

import sinabs.layers as sl
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork, DynapcnnNetwork
from sinabs.backend.dynapcnn.dynapcnn_layer import DynapcnnLayer


def to_tuple(x):
    return (x, x) if isinstance(x, int) else x


def convert_linear_to_convolutional(
    layer: nn.Linear, input_shape: Tuple[int, int, int]
) -> nn.Conv2d:
    """Converts a linear layer to a convolutional layer. For original refer to:
    sinabs/backend/dynapcnn/dynapcnn_layer.py::DynapcnnLayer::_convert_linear_to_conv.

    Args:
        layer (nn.Linear): A linear layer
        input_shape (Tuple[int, int, int]): Input shape in (channel, y, x)

    Returns:
        nn.Conv2d: A convolutional layer
    """
    in_channels, in_height, in_width = input_shape

    if layer.in_features != in_channels * in_height * in_width:
        raise ValueError(
            f"Linear layer has {layer.in_features} features."
            + f"However, the input to network has {in_channels * in_height * in_width}"
        )

    conv_layer = nn.Conv2d(
        in_channels=in_channels,
        kernel_size=(in_height, in_width),
        out_channels=layer.out_features,
        padding=0,
        bias=False,
    )
    conv_layer.weight.data = (
        layer.weight.data.clone()
        .detach()
        .reshape((layer.out_features, in_channels, in_height, in_width))
    )
    return conv_layer


def convert_convolutional_layer(
    layer: nn.Conv2d, input_shape: Tuple[int, int, int], weight_scale: float = 1.0
) -> Tuple[ConvFilter, Tuple[int, int, int]]:
    """Convert a convolutional layer to samna filter.

    Args:
        layer (nn.Conv2d): A PyTorch Convolutional Layer. The biases has to be disabled.
        input_shape (Tuple[int, int, int]): Input shape of the layer in (channel, y, x)
        weight_scale (float): Multiply the layer weights. This is often necessary when converting
        models with AvgPool2d

    Returns:
        Tuple[ConvFilter, Tuple[int, int, int]]: Returns the samna filter and the output shape
        of the layer
    """
    if layer.bias is not None:
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
        in_channels, out_channels, kernel_size, in_shape, stride, padding
    )

    # Set the layer weights
    weights = layer.weight.cpu().detach().clone()
    weights /= weight_scale
    conv_filter.set_weights(weights.tolist())

    # Get the output image shape
    output_shape = [out_channels, *conv_filter.get_layer().get_output_shape()]

    return conv_filter, output_shape


def convert_pooling_layer(
    layer: Union[nn.AvgPool2d, sl.SumPool2d], input_shape: Tuple[int, int, int]
) -> Tuple[SumPoolFilter, Tuple[int, int, int]]:
    """Converts a pooling layer to a samna filter.

    Args:
        layer (Union[nn.AvgPool2d, sinabs.layers.SumPool2d]): A pooling layer.
        input_shape (Tuple[int, int, int]): Input shape to the pooling layer in (channel, y, x)

    Returns:
        Tuple[SumPoolFilter, Tuple[int, int, int]]: Returns a tuple of sum pooling filter and
        the output shape in (channel, y, x)
    """

    ## Keeping this commented out due to missing support to DynapcnnLayer conversion. However,
    ## this check is the correct one.
    # if to_tuple(layer.kernel_size) != to_tuple(layer.stride):
    #     raise ValueError("For pooling layers kernel size has to be the same as the stride")

    # initialize filter
    pooling_filter = SumPoolFilter()

    # extract parameters from layer
    kernel_size = to_tuple(layer.kernel_size)
    in_shape = tuple(input_shape[1:])

    # set the filter parameters
    pooling_filter.set_parameters(kernel_size, in_shape)

    # get the output shape
    output_shape = [input_shape[0], *pooling_filter.get_layer().get_output_shape()]

    return pooling_filter, output_shape


def convert_iaf_layer(
    layer: Union[sl.IAF, sl.IAFSqueeze], input_shape: Tuple[int, int, int]
) -> Tuple[IAFFilter, Tuple[int, int, int]]:
    """Convert a sinabs IAF layer into a specksim IAF Filter.

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
    iaf_filter.set_parameters(n_channels, in_shape, spike_threshold, min_v_mem)
    return iaf_filter, input_shape


def calculate_weight_scale(layer: nn.AvgPool2d):
    """Calculate the weight scale for the next weight layer given an AvgPool layer. This is
    necessary, because only real supported pooling layer is SumPooling for the simulator.

    Args:
        layer (nn.AvgPool2d): torch Average pooling layer.
    """
    kernel_size = to_tuple(layer.kernel_size)

    # # Keeping this commented out due to missing support to DynapcnnLayer conversion. However,
    # # this check is the correct one.
    # stride = to_tuple(layer.kernel_size)
    # if kernel_size != stride:
    #     raise ValueError("Kernel size and stride of the Average pooling layer should be the same.")

    # calculate and return the weight scale based on kernel size
    # this effectively converts an average pooling layer to a sum pooling
    weight_scale = float(kernel_size[0] * kernel_size[1])
    return weight_scale


def from_sequential(
    network: nn.Sequential, input_shape: Tuple[int, int, int]
) -> "SpecksimNetwork":
    """Convert a sinabs network to a SpecksimNetwork.

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
    current_weight_scale: float = 1.0

    # Add an input node
    filters.append(samna.BasicSourceNode_specksim_events_spike())

    for name, layer in network.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            if isinstance(layer, nn.Linear):
                layer = convert_linear_to_convolutional(layer, current_shape)
            samna_filter, current_shape = convert_convolutional_layer(
                layer, current_shape, current_weight_scale
            )
            current_weight_scale = 1.0  # set the weight scale back to normal
        elif isinstance(layer, (sl.SumPool2d, nn.AvgPool2d)):
            samna_filter, current_shape = convert_pooling_layer(layer, current_shape)
            if isinstance(layer, nn.AvgPool2d):
                current_weight_scale = calculate_weight_scale(layer)
        elif isinstance(layer, (sl.IAF, sl.IAFSqueeze)):
            samna_filter, current_shape = convert_iaf_layer(layer, current_shape)
        elif isinstance(layer, nn.Flatten):
            continue
        elif isinstance(
            layer,
            (nn.Sequential, DynapcnnLayer, DynapcnnNetwork, DynapcnnCompatibleNetwork),
        ):
            continue  # Do not issue errors for these classes.
        elif isinstance(layer, nn.ReLU):
            raise TypeError(
                f"ReLU layer with name: {name} found!"
                + "Please convert your model to a spiking model before converting to specksim."
            )
        else:
            warn(
                f"Layer with name: {name} of type: {type(layer)} is ignored"
                + "and will not be included. Your network may not be properly simulated."
            )
            continue

        filters.append(samna_filter)

    # Add an output node
    filters.append(samna.BasicSinkNode_specksim_events_spike())
    members = graph.sequential(filters)

    return SpecksimNetwork(graph, members)


class SpecksimNetwork:
    output_dtype = np.dtype(
        [("x", np.uint32), ("y", np.uint32), ("t", np.uint32), ("p", np.uint32)]
    )

    def __init__(
        self,
        graph: samna.graph.EventFilterGraph,
        graph_members: List["SamnaFilterNode"],  # noqa: F821
        initial_sleep_duration: float = 1.0,
        subsequent_sleep_duration: float = 0.1,
    ):
        """Specksim simulation container object.

        Args:
            graph (samna.graph.EventFilterGraph): A samna graph that contains the network layers as samna filters.
            graph_members (List["SamnaFilterNode"]): A list of samna filters.
            initial_sleep_duration (float): Sleep between writing and reading from the samna graph structure. This is
            needed because the graph runs on a separate thread.
            subsequent_sleep_duration (float): In order to not drop any events, we can sleep for more time.
        """
        self.network: samna.graph.EventFilterGraph = graph
        self.members = graph_members
        self.initial_sleep_duration = initial_sleep_duration
        self.subsequent_sleep_duration = subsequent_sleep_duration

        # Monitor mechanics
        self.monitors: Dict[int, Dict[str, List]] = {}

    def forward(self, xytp: np.record) -> np.record:
        """Applies the network forward pass given events.

        Args:
            xytp (np.record): Input events as a numpy record array with keys ('x', 'y', 't', 'p')

        Returns:
            np.record: Output events as a numpy record array of the same type as the input events.
        """
        # set up container for output spikes
        output_spikes = []
        # convert xytp to specksim spikes
        spikes = self.xytp_to_specksim_spikes(xytp)
        # start the network graph
        self.network.start()

        # start the monitor graph(s)
        for monitor in self.monitors.values():
            # flush the monitor buffers.
            _ = monitor["sink"].get_events()
            monitor["graph"].start()

        # do the forward pass
        self.members[0].write(spikes)  # write
        time.sleep(self.initial_sleep_duration)
        output_spikes = self.members[-1].get_events()  # read

        # check if any events are produced after reading
        while True:
            time.sleep(self.subsequent_sleep_duration)
            previous_spike_count = len(output_spikes)
            output_spikes.extend(self.members[-1].get_events())
            current_spike_count = len(output_spikes)
            if current_spike_count == previous_spike_count:
                break

        # stop the monitor graph(s)
        for monitor in self.monitors.values():
            monitor["graph"].stop()

        # stop the streaming graph at the end
        self.network.stop()

        return self.specksim_spikes_to_xytp(output_spikes, self.output_dtype)

    def __call__(self, xytp: np.record) -> np.record:
        return self.forward(xytp)

    def reset_states(self):
        """
        Reset the states of every spiking layer in the network to 0.
        """
        for member in self.members:
            if isinstance(member, IAFFilter):
                member.get_layer().reset_states()

    def get_nth_spiking_layer(self, spike_layer_number: int) -> IAFFilter:
        """Get nth spiking layer for reading.

        Args:
            spike_layer_number (int): `spike_layer_number`th IAFFilter

        Returns:
            IAFFilter: `spike_layer_number`th IAFFilter
        """
        spike_layer_idx = 0
        for member in self.members:
            if isinstance(member, IAFFilter):
                if spike_layer_idx == spike_layer_number:
                    return member
                spike_layer_idx += 1

        raise ValueError(f"{spike_layer_number}th monitor does not exist!")

    def add_monitor(self, spike_layer_number: int):
        """Add a monitor to the `spike_layer_number`th IAF layer.

        Args:
            spike_layer_number (int): `spike_layer_number`th IAF layer to monitor
        """
        iaf_filter = self.get_nth_spiking_layer(spike_layer_number)
        graph = samna.graph.EventFilterGraph()
        _, sink = graph.sequential(
            [iaf_filter, samna.BasicSinkNode_specksim_events_spike()]
        )
        self.monitors.update({spike_layer_number: {"graph": graph, "sink": sink}})

    def add_monitors(self, spike_layer_numbers: List[int]):
        """Convenience function to add monitor to multiple spike layers.

        Args:
            spike_layer_numbers (List[int]): Numbers of the spike spike layers to monitor.
        """
        for number in spike_layer_numbers:
            self.add_monitor(number)

    def read_monitor(self, spike_layer_number: int) -> np.record:
        """Read the events from the `spike_layer_number`th hidden spiking layer.

        Args:
            spike_layer_number (int): `spike_layer_number`th spiking layer to monitor.

        Returns:
            np.record: Events from `spike_layer_number`th spiking layer as a numpy
                record array.
        """
        if spike_layer_number not in self.monitors.keys():
            raise ValueError(f"Spike layer: {spike_layer_number} is not ")
        return self.specksim_spikes_to_xytp(
            self.monitors[spike_layer_number]["sink"].get_events_blocking(),
            self.output_dtype,
        )

    def read_monitors(self, spike_layer_numbers: List[int]) -> Dict[int, np.record]:
        """Convenience method to read from multiple monitors.

        Args:
            spike_layer_numbers (List[int]): a list of spike layer numbers.

        Returns:
            Dict[int, np.record]: Dict with keys of spike_layer_numbers and events in np.record
                format with 4 keys `x`, `y`, `p`, `t`
        """
        spike_dict: Dict[int, np.record] = {}
        for number in spike_layer_numbers:
            spike_dict.update({number: self.read_monitor(number)})
        return spike_dict

    def read_all_monitors(self):
        """Convenience method to read all the monitors."""
        spike_dict: Dict[int, np.record] = {}
        for number in self.monitors.keys():
            spike_dict.update({number: self.read_monitor(number)})
        return spike_dict

    def read_spiking_layer_states(
        self, spike_layer_number: int
    ) -> List[List[List[int]]]:
        """Read the states of the `spike_layer_number`th spiking layer.

        Args:
            spike_layer_number (int): `spike_layer_number`th spiking layer to read states from.

        Returns:
            List[List[List[int]]]: 3-dimensional list of states in (channel, y, x)
        """
        iaf_filter = self.get_nth_spiking_layer(spike_layer_number)
        return iaf_filter.get_layer().get_v_mem()

    def clear_monitors(self):
        """Clear all monitors."""
        self.monitors = {}

    @staticmethod
    def xytp_to_specksim_spikes(xytp: np.record) -> List[samna.specksim.events.Spike]:
        """Takes in xytp and returns a list of spikes compatible with specksim.

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
    def specksim_spikes_to_xytp(
        spikes: List[samna.specksim.events.Spike], output_dtype: np.dtype
    ) -> np.record:
        """Takes in specksim spikes and converts them to record array of with keys "x", "y", "t"
        and "p".

        Args:
            spikes (List[samna.specksim.events.Spike]): A list of specksim spikes coming from the output
            of the network.
            output_dtype: type of the output spikes. This is defined in the class implementation.

        Returns:
            np.record: A record array of the given output type
        """
        output_events = []
        for spike in spikes:
            x, y, t, p = spike.x, spike.y, spike.timestamp, spike.feature
            output_event = (x, y, t, p)
            output_events.append(output_event)
        return np.array(output_events, dtype=output_dtype)
