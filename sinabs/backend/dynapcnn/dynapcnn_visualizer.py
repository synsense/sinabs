import samna, samnagui
from multiprocessing import Process
from typing import List, Tuple, Union, Optional, Dict
import os, socket
import time
import warnings
from functools import partial
from sinabs.backend.dynapcnn.dynapcnn_network import DynapcnnNetwork

class DynapcnnVisualizer:
    """
    cluster_dvs_layout = (0, 0, 0.33, 1)
    cluster_readout_layout = (0.66, 0, 1, 1)
    cluster_spike_count_layout = (0.33, 0, 0.66, 1)
    (tlx, tly, brx, bry)
    """
    # Default layouts
    # (tlx, tly, brx, bry)
    DEFAULT_LAYOUT_DS = [
        (0, 0, 0.5, 1), (0.5, 0, 1, 1), None, None
    ]
    DEFAULT_LAYOUT_DSP = [
        (0, 0, 0.5, 0.66), (0.5, 0, 1, 0.66), None, (0, 0.66, 1, 1)
    ]
    DEFAULT_LAYOUT_DSR = [
        (0, 0, 0.33, 1), (0.33, 0, 0.66, 1), (0, 0.66, 1, 1), None
    ]
    
    DEFAULT_LAYOUT_DSRP = [
        (0, 0, 0.33, 0.66), (0.33, 0, 0.66, 0.66), (0.66, 0, 1, 0.66), (0, 0.66, 1, 1)
    ]

    LAYOUTS_DICT = {
        "ds": DEFAULT_LAYOUT_DS,
        "dsp": DEFAULT_LAYOUT_DSP,
        "dsr": DEFAULT_LAYOUT_DSR,
        "dsrp": DEFAULT_LAYOUT_DSRP
    }

    def __init__(
        self,
        dvs_shape: Tuple[int, int] = (128, 128),  # height, width
        add_readout_plot: bool = False,
        add_power_monitor_plot: bool = False,
        spike_collection_interval: int = 500,
        readout_prediction_threshold: int = 10,
        readout_default_return_value: Optional[int] = None,
        feature_names: Optional[List[str]] = None, 
        readout_images: Optional[List[str]] = None,
        feature_count: Optional[int] = None
    ):
        """Quick wrapper around Samna objects to get a basic dynapcnn visualizer.

        Args:
            dvs_shape (Tuple[int, int], optional): 
                Shape of the DVS sensor in (height, width). 
                Defaults to (128, 128) -- Speck sensor resolution.
            add_readout_plot: bool (defaults to False)
                If set true adds a readout plot to the GUI
            add_power_monitor_plot: bool (defaults to False)
                If set true adds a power monitor plot to the GUI.
            spike_collection_interval: int (defaults to 500) (in milliseconds)
                Spike collection is done using a low-pass filter with a window size.
                This parameter sets the window size of the spike collection 
            readout_prediction_threshold: int (defaults to 10)
                Defines the number of spikes needed for making a prediction.
            readout_default_return_value: Optional[int] (defaults to None)
                Defines the default prediction of the network. Usually used for `other` class in the
                network.
            feature_names: Optional[List[str]] (defaults to None)
                List of feature names. If this is passed they will be displayed on the spike count plot
                as output labels
            readout_images: Optional[List[str]] (defaults to None)
                List of paths of the images to be shown in the readout plot. 
                If the `feature_names` parameter is not passed the names of the images will be parsed and
                used as the spike count plot labels.
                Format of the individual file name should be of the following type.
                `classnumber`_`classlabel`.`extension`
                NOTE: Class numbers should match that of the network output channels. This is so that they
                can be sorted properly. At each operating system the behaviour in which the extraction of the
                images from a folder may differ.
            feature_count: Optional[int] (defaults to None)
                If the `feature_names` and `readout_images` was passed, this is not needed. Otherwise this parameter
                should be passed, so that the GUI knows how many lines should be drawn on the `Spike Count Plot` and
                `Readout Layer Plot`. 

        """
        # Visualizer layout components
        self.feature_names = feature_names
        self.readout_images = readout_images
        self.feature_count = feature_count
        self.dvs_shape = dvs_shape

        # Modify the GUI type based on the parameters
        self.gui_type = "ds" 
        if add_readout_plot:
            self.gui_type += "r"
        if add_power_monitor_plot:
            self.gui_type += "p"

        # Spike count layer components
        self.spike_collection_interval = spike_collection_interval

        # Readout layer components
        self.readout_prediction_threshold = readout_prediction_threshold
        self.readout_default_return_value = readout_default_return_value

        # Samna TCP communication ports
        ## Default samna ports
        self.samna_receiver_port = "33335"
        self.samna_sender_port = "33336"
        ## Visualizer port
        self.samna_visualizer_port = get_free_tcp_port()

        # Initialize samna node for visualization
        self.samna_node = samna.init_samna(f"tcp://0.0.0.0:{self.samna_receiver_port}", f"tcp://0.0.0.0:{self.samna_sender_port}")
        time.sleep(1)
    
    @staticmethod
    def parse_feature_names_from_image_names(
        readout_image_paths: List[str] 
    ):
        """Method the parse the feature names directly from the names of the images.
        Args:
            readout_image_paths: List[str] 
                List of paths to all the feature images
        """
        if readout_image_paths:
            feature_names = []
            for path in readout_image_paths:
                image_name = path.split("/")[-1] 
                image_name = image_name.split("_")[-1] # split number, name
                image_name = image_name.split(".")[0] # split name, extension
                feature_names.append(image_name)
            return feature_names

        else:
            return None
    
    def create_visualizer_process(
        self,
        initial_window_scale: Tuple[int, int] = (2, 4),
        visualizer_id: int = 3
    ):
        """Create a samnagui visualizer process

        Args:
            sender_endpoint (str): 
                Samna node sender endpoint
            receiver_endpoint (str): 
                Samna node receiver endpoint
            initial_window_scale (Tuple[int, int], optional): 
                Initial scale of one visualizer window in (height, width). Defaults to (2, 4).
            visualizer_id (int, optional): 
                Id of the visualizer node to be created. 
                Defaults to 3. -- samna default 

        Returns:
            samna.submodule: 
                Samna remote visualizer node.
        """
        # Height and width based on the proportion to a 16:9 screen.
        height_proportion = 1/9 * initial_window_scale[0]
        width_proportion = 1/16 * initial_window_scale[1]
        
        # Create and start the process
        process = Process(
            target=samnagui.runVisualizer,
            args=(
                width_proportion,
                height_proportion,
                self.samna_node.get_receiver_endpoint(),
                self.samna_node.get_sender_endpoint(),
                visualizer_id
            )
        )                
        process.start()
        
        # Wait until the process is properly initialized
        time.sleep(2)
        
        # Create a remote samna node and return it.
        samna.open_remote_node(visualizer_id, f"visualizer_{visualizer_id}")
        return getattr(samna, f"visualizer_{visualizer_id}")
    
    
    def add_dvs_plot(
        self, 
        visualizer,
        shape: Tuple[int, int], 
        layout: Tuple[float, float, float, float],
    ):
        """Add an activity plot (dvs plot) to a visualizer

        Args:
            visualizer (samna.submodule):
                Remote samnagui node.
            shape (Tuple(int, int)):
                Shape of the plot in (height, width) 
            layout (Tuple[float, float, float, float]): 
                Layout to position the plot on the samnagui visualizer in
                (top-left-x, top-left-y, bottom-right-x, bottom-right-y)

        Returns:
            Tuple[samna.ui.ActivityPlot, int]:
                A tuple of the plot object and its id.
        """
        plot_id = visualizer.plots.add_activity_plot(shape[1], shape[0], "Dvs Plot")
        plot = getattr(visualizer, f"plot_{plot_id}")
        plot.set_layout(*layout)
        return (plot, plot_id)

    def add_readout_plot(
        self,
        visualizer,
        layout: Tuple[float, float, float, float],
        images: List[str]
    ):
        """Add a readout plot (image showing the predicted class) to the visualizer 

        Args:
            visualizer (samna.submodule): 
                Remote samnagui node.
            layout (Tuple[float, float, float, float]): 
                Layout to position the plot on the samnagui visualizer in
                (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
            images (List[str]): 
                A list of paths to the images corresponding to be shown in the case that the class
                is predicted. -- Note that order of this list should match the order of the lookup
                table.

        Returns:
            Tuple[samna.ui.ReadoutPlot, int]: 
                A tuple of the plot object and its id.
        """
        plot_id = visualizer.plots.add_readout_plot(
            "Readout Plot",
            images
        )
        plot = getattr(visualizer, f"plot_{plot_id}")
        plot.set_layout(*layout)
        return (plot, plot_id)
    
    def add_output_prediction_layer_plot():
        """
        What we want to have is something as described below:
        Plot for visualizating the chip readout layers.

        output neuron id
        ^ 
        | 
        +-------+----------------------+
        + out0  +           x          +
        + out1  +     x                +
        + out2  +   xxxxxxxx    xxxxx  +
        +-------+----------------------+ --> time

        Where every time the readout layer has been read, if some output returns True, put an `x` there,
        denoting a prediction
        """
        raise NotImplementedError("This is not implemented as it has to be supported by samna first!")
    
    def add_spike_count_plot(
        self, 
        visualizer,
        layout: Tuple[float, float, float, float],
        **kwargs
    ):
        """Add a spike count plot (line plot showing recent predicitons from network 
            for each class)

        Args:
            visualizer (samna.submodule):
                Remote samnagui node.
            layout (Tuple[float, float, float, float]): 
                Layout to position the plot on the samnagui visualizer in
                (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
            kwargs:
                TODO: Support all kwargs

        Returns:
            Tuple[samna.ui.SpikeCountPlot, int]: 
                A tuple of the plot object and its id.
        """
        plot_id = visualizer.plots.add_spike_count_plot(
            "Spike Count Plot",
            self.feature_count,
            self.feature_names
        )
        spike_count_plot = getattr(visualizer, f"plot_{plot_id}")
        spike_count_plot.set_layout(*layout)
        spike_count_plot.set_show_x_span(25)
        spike_count_plot.set_label_interval(2.5)
        spike_count_plot.set_max_y_rate(1.2)
        spike_count_plot.set_show_point_circle(True)
        spike_count_plot.set_default_y_max(10)
        # Check if the method name and value are in kwargs
        for k, v in kwargs.items():
            try:
                method = getattr(spike_count_plot, k)
                method(v)
            except:
                warnings.warn(f"The passed keyword does not exist in `Spike Count Plot`: {k}")
        return (spike_count_plot, plot_id)
    
    def add_power_monitor_plot(
        self, 
        visualizer,
        layout: Tuple[int, int, int, int],
        **kwargs
    ):
        """
        Args:
            visualizer (samna.submodule):
                Remote samnagui node.
            layout (Tuple[float, float, float, float]): 
                Layout to position the plot on the samnagui visualizer in
                (top-left-x, top-left-y, bottom-right-x, bottom-right-y)
            kwargs:
                TODO: Support all kwargs

        Returns:
            Tuple[samna.ui.PowerMeasurementPlot, int]: 
                A tuple of the plot object and its id.
        """

        # Set up power measurement plot
        power_measurement_plot_id = visualizer.plots.add_power_measurement_plot(
            "Power monitor plot",
            3,
            ["io", "ram", "logic"]
        )
        power_measurement_plot = getattr(visualizer, f"plot_{power_measurement_plot_id}")
        power_measurement_plot.set_layout(*layout)
        power_measurement_plot.set_show_x_span(10)
        power_measurement_plot.set_label_interval(2)
        power_measurement_plot.set_max_y_rate(1.5)
        power_measurement_plot.set_show_point_circle(False)
        power_measurement_plot.set_default_y_max(1)
        power_measurement_plot.set_y_label_name("power (mW)")
        # Check if the method name and value are in kwargs
        for k, v in kwargs.items():
            try:
                method = getattr(power_measurement_plot, k)
                method(v)
            except:
                warnings.warn(f"The passed keyword does not exist in `Power Monitor Plot`: {k}")

    def create_plots(
            self, 
            visualizer_node
        ):
            """Utility function to create a Cluster visualizer

            Args:
                visualizer (samna.submodule):
                    Remote samnagui node.

            Returns:
                Tuple[Tuple[samna.ui.Plot, int]]: 
                    A tuple of tuples of the plot objects and their ids.
            """
            layout = self.LAYOUTS_DICT[self.gui_type]

            dvs_plot = self.add_dvs_plot(
                visualizer_node, 
                shape=self.dvs_shape, 
                layout=layout[0] 
            )
            spike_count_plot = self.add_spike_count_plot(
                visualizer_node, 
                layout=layout[1]
            )
            if "r" in self.gui_type:
                try:
                    readout_plot = self.add_readout_plot(
                    visualizer_node,
                    layout=layout[2],
                    images=self.readout_images
                    )
                except:
                    readout_plot = None
                    print(f"Either the layout or the images are missing in the readout plot. ")
            else:
                readout_plot = None
            
            if "p" in self.gui_type:
                try:
                    power_monitor_plot = self.add_power_monitor_plot(visualizer_node, layout=layout[3])
                except:
                    power_monitor_plot = None
                    print(f"Layout missing the power monitor plot. ")
            else:
                readout_plot = None

            
            return dvs_plot, spike_count_plot, readout_plot, power_monitor_plot
    
    def connect(
        self, 
        dynapcnn_network: DynapcnnNetwork
    ):
        # Checks for the visualizer to work correctly.
        if not hasattr(dynapcnn_network, "samna_device"):
            raise ConnectionError(
                "Model has to be ported to chip.\n" + 
                "Hint: Call `.to()` method on the model. "
            )
        
        config = dynapcnn_network.samna_config
        if not config.dvs_layer.monitor_enable:
            raise ValueError(
                "DVS layer is not monitored.\n" + 
                "Hint: in `.to()` method `monitor_layers` parameter" + 
                " should contain value `dvs`. "
            )
        
        last_layer = dynapcnn_network.chip_layers_ordering[-1]

        if not config.cnn_layers[last_layer].monitor_enable:
            raise ValueError(
                f"Last layer running on core: {last_layer} is not monitored." +
                "Hint: in `.to()` method `monitor layers` parameter should " +
                "contain key `-1` or the last layer `idx`. "
            )
        
        # Update the feature count before initializing and connecting plots
        if self.feature_count is None:
            self.update_feature_count(dynapcnn_network)
        
        if self.feature_names is None:
            self.update_feature_names()
        
        # Create graph and connect network attributes to plots
        ## Determine the port and create the graph
        self.streamer_graph = samna.graph.EventFilterGraph()

        ## Start visualizer and create plots based on parameters.
        remote_visualizer_node = self.create_visualizer_process(
            initial_window_scale=(2, 4),
            visualizer_id=3
        )

        (dvs_plot, spike_count_plot, readout_plot, power_plot) = self.create_plots(visualizer_node=remote_visualizer_node)

        # Streamer graph
        # Dvs node
        (_, event_type_filter, _, streamer_node) = self.streamer_graph.sequential([
            dynapcnn_network.samna_device.get_model_source_node(),
            samna.graph.JitDvsEventToViz(), # Here goes JitDvsToViz()
            "VizEventStreamer"
        ])

        ## Spike count node
        (
            _, 
            member_filter, 
            spike_collection_node, 
            spike_count_node, 
            streamer_node
        ) = self.streamer_graph.sequential([
            dynapcnn_network.samna_device.get_model_source_node(),
            samna.graph.JitMemberSelect(),
            samna.graph.JitSpikeCollection(),
            samna.graph.JitSpikeCount(),
            streamer_node 
        ])
        member_filter.set_white_list([last_layer], "layer")
        spike_collection_node.set_interval_milli_sec(self.spike_collection_interval)
        spike_count_node.set_feature_count(self.feature_count)

        ## Power monitor node
        if "p" in self.gui_type:
            # Initialize power monitor
            power_monitor = dynapcnn_network.samna_device.get_power_monitor()
            power_monitor.start_auto_power_measurement(50)
            # Connect its output to streamer_node
            self.streamer_graph.sequential([
                power_monitor.get_source_node(),
                "MeasurementToVizConverter",
                streamer_node
            ])
        
        ## Readout node
        if "r" in self.gui_type:
            (_, majority_readout_node, _) = self.streamer_graph.sequential([ 
                spike_collection_node, 
                samna.graph.MajorityReadout(),
                streamer_node 
            ])
            majority_readout_node.set_feature_count(self.feature_count)
            majority_readout_node.set_default_feature(self.readout_default_return_value)
            # TODO: Handle the following parameters
            # majority_readout_node.set_threshold_low()
            # majority_readout_node.set_threshold_high()
        
        ## Readout layer visualization
        if "o" in self.gui_type:
            raise NotImplementedError("A lot of work in progress!")
        
        streamer_node.set_streamer_endpoint(f"tcp://0.0.0.0:{self.samna_visualizer_port}")
        
        # Connect the visualizer components
        remote_visualizer_node.receiver.set_receiver_endpoint(f"tcp://0.0.0.0:{self.samna_visualizer_port}")
        remote_visualizer_node.receiver.add_destination(remote_visualizer_node.splitter.get_input_channel())
        ## Connect Dvs visualizer to its plot
        remote_visualizer_node.splitter.add_destination("dvs_event", remote_visualizer_node.plots.get_plot_input(dvs_plot[1]))
        ## Connect Spike count visualizer to its plot
        remote_visualizer_node.splitter.add_destination("spike_count", remote_visualizer_node.plots.get_plot_input(spike_count_plot[1]))
        ## Connect Readout visualizer to its plot
        if "r" in self.gui_type:
            remote_visualizer_node.splitter.add_destination("readout", remote_visualizer_node.plots.get_plot_input(readout_plot[1]))
        ## Connect power monitor visualizer to its plot
        if "p" in self.gui_type:
            remote_visualizer_node.splitter.add_destination("measurement", remote_visualizer_node.plots.get_plot_input(power_plot[1]))


    def update_feature_count(self, dynapcnn_network: DynapcnnNetwork):
        # Find last layer output
        raise NotImplementedError("Work in progress!") 
    
    def update_feature_names(self):
        if self.readout_images:
            self.feature_names = self.parse_feature_names_from_image_names(readout_image_paths=self.readout_images)
        else:
            self.feature_names = [f"{i}" for i in range(len(self.feature_count))]
    
    def start(self):
        self.streamer_graph.start()

    def stop(self):
        self.streamer_graph.stop()


def get_free_tcp_port():
    """Returns a free tcp port.
    Returns:
        str: A port which is free in the system 
    """
    free_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_socket.bind(('0.0.0.0', 0))
    free_socket.listen(5)
    port = free_socket.getsockname()[1] # get port
    free_socket.close()
    return port