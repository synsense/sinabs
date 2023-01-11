import samna, samnagui
from multiprocessing import Process
from typing import List, Tuple, Union, Optional, Dict
import os
import time
import warnings
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
        sender_endpoint: str,
        receiver_endpoint: str,
        dvs_shape: Tuple[int, int] = (128, 128),  # height, width
        gui_type: str = "ds",
        feature_names: Optional[List[str]] = None, 
        readout_images: Optional[List[str]] = None,
        feature_count: Optional[int] = None
    ):
        """Quick wrapper around Samna objects to get a basic dynapcnn visualizer.

        Args:
            sender_endpoint (str): 
                Samna node sender endpoint 
            receiver_endpoint (str):
                Samna node receiver endpoint
            dvs_shape (Tuple[int, int], optional): 
                Shape of the DVS sensor in (height, width). 
                Defaults to (128, 128) -- Speck sensor resolution.
            gui_type: str (defaults to "ds")
                Which GUI components are required.
                Options:
                    "ds"   -> Dvs plot + Spike count plot
                    "dsp"  -> Dvs plot + Spike count plot + power monitor plot
                    "dsr"  -> Dvs plot + Spike count plot + readout plot
                    "dsrp" -> Dvs plot + Spike count plot + readout plot + power monitor plot 
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
        # Samna components
        self.receiver_endpoint = receiver_endpoint
        self.sender_endpoint = sender_endpoint
        
        # Visualizer layout components
        self.feature_names = feature_names
        self.readout_images = readout_images
        self.feature_count = feature_count
        self.dvs_shape = dvs_shape
        self.gui_type = gui_type
    
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
    
    @staticmethod
    def create_visualizer_process(
        sender_endpoint: str,
        receiver_endpoint: str,
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
                receiver_endpoint,
                sender_endpoint,
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
    
    def add_readout_layer_plot():
        """
        What we want to have is something as described below:

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
        feature_names = list(self.lookup_table.values())
        feature_count = len(feature_names)
        plot_id = visualizer.plots.add_spike_count_plot(
            "Spike Count Plot",
            feature_count,
            feature_names
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
                "contain key `-1`. "
            )
        
        # Create graph and connect network attributes to plots

        ## Start visualizer and create plots based on parameters.
        remote_visualizer_node = self.create_visualizer_process(
            sender_endpoint=...,
            receiver_endpoint=...,
            initial_window_scale=...,
            visualizer_id=...
        )

        (dvs_plot, spike_count_plot, readout_plot, power_plot) = self.create_plots(visualizer_node=remote_visualizer_node)

        ## Create graph
        self.communication_graph = samna.graph.EventFilterGraph()
