# Dynapcnn Visualizer

## Introduction
Currently the development kits that SynSense provides are mainly used for benchmarking purposes. However, through our backend library `samna` we support processing of events and interpretation of the output in a streaming fashion. `samna` API however is mainly designed and developed for low-level communication with the chips. This sometimes makes it tricky to work with for higher-level functions. `samna` also has a package called `samnagui` with which we can do some visualization. Internally, we use it for testing of our models in real-time and real-life conditions. 
In order to give users an easy access without having to deal with a lot of boilerplate code and burdensome logic, we implemented a visualizer class.  

## Available plots in `samnagui`
There are 4 different plots in `samnagui` that are available for use.
- <b>Activity Plot:</b> A plot that visualizes the events produced by the on-chip sensor.
- <b>Line Plot:</b> A plot that allows visualization of the events produced by the model running on the chip. Additionally this is used when we display power consumption measurements.
- <b>Image Plot:</b> A plot that allows you to display an image. Internally we have used this plot to display an image denoting a predicted output class. 

Documentation is available under the following link: [samna.ui documentation](https://synsense-sys-int.gitlab.io/samna/reference/ui/index.html)

## Useful `samna` nodes for visualizing and Just-In-Time (JIT) compiled nodes.
### Nodes
`samna` also comes with implementation of several nodes that are useful for communicating between a `samnagui` visualizer and the chip. <br>
Events that are related to DVS events from the sensor are:
- `DvsEventDecimate`: Eliminating `L` out of `M` events. <br>
<t>`set_decimation_fraction(M: int, L: int)`
- `DvsEventRescale`: Rescale events. `x / width` and `y / height`. <br>
<t>`set_rescaling_coefficients(width_coeff: int, height_coeff: int)`
- `DvsToVizConverter`: Converts events from sensor to visualization events. 
<br>The ones that are connected to the chip output spikes:
- `SpikeCollectionNode`: Picks output spikes at intervals (ms) and makes events that can be used out of them. <br>
<t>`set_interval_milli_sec(interval: int)`
- `SpikeCountNode`: Counts how many events from each output received among `feature_count` events and outputs a visualizer event. <br>
<t>`set_feature_count(feature_count: int)`
- `MajorityReadoutNode`: Among the events produced by `SpikeCollectionNode` selects the most active output channel. Used alongside the `ImagePlot`.

These nodes are available for different chips. Further documentation can be found in the following link: [samna-available-filters](https://synsense-sys-int.gitlab.io/samna/filters.html) 

### Just-In-Time compiled nodes
The nodes mentioned above are also available for any devboard under `samna.graph.Jit{nameOfNode}`. 

## Dynapcnn Visualizer
`DynapcnnVisualizer` class uses the nodes and plots mentioned above in order to use it and provides an intuitive interface. An example code-snippet can be found below:

#### Setup
```
import os
current_folder_path = str(os.path.join(os.getcwd()))
file_tokens = current_folder_path.split("/")[:-3]
params_path = os.path.join( os.path.join("/", *file_tokens), "examples/dvs_gesture_params.pt") 
icons_folder_path = os.path.join( os.path.join("/", *file_tokens), "examples/icons/")
```

#### Import requirements
```
import torch
import torch.nn as nn
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork
```

#### Define model
```
ann = nn.Sequential(
    nn.Conv2d(2, 16, kernel_size=2, stride=2, bias=False),
    nn.ReLU(),
    # core 1
    nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # core 2
    nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # core 7
    nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # core 4
    nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # core 5
    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    # core 6
    nn.Dropout2d(0.5),
    nn.Conv2d(64, 256, kernel_size=2, bias=False),
    nn.ReLU(),
    # core 3
    nn.Dropout2d(0.5),
    nn.Flatten(),
    nn.Linear(256, 128, bias=False),
    nn.ReLU(),
    # core 8
    nn.Linear(128, 11, bias=False),
)
```

#### Load model weights from the example folder

```
load_result = ann.load_state_dict(torch.load(params_path), strict=False)
```

#### Convert to SNN
When converting to a `DynapcnnNetwork` using `sinabs` and `sinabs-dynapcnn`, please use the following flags:
- `dvs_input`=`True` has to be set so that the model can receive input from the on-board sensor or an external DVS sensor.
- `discretize`=`True` has to be set so that the model can be ported to the chip.

```
sinabs_model = from_model(ann, add_spiking_output=True, batch_size=1)

input_shape = (2, 128, 128)
hardware_compatible_model = DynapcnnNetwork(
    sinabs_model.spiking_model.cpu(),
    dvs_input=True,
    discretize=True,
    input_shape=input_shape
)
```

#### Port the model to the chip
Port the model on the chip. Important! <br>
- `monitor_layers` = `["dvs", -1]` Other layers can also be monitored for other purposes, but `monitor_layers` should contain at least `dvs` and `-1` so that the visualizer has access to the `dvs` layer and `-1` for the output layer of the network. 

```
hardware_compatible_model.to(
    device="speck2edevkit", # speck2edevkit
    monitor_layers=["dvs", -1],  # Last layer
    chip_layers_ordering="auto"
)
```

#### Use DynapcnnVisualizer
In order to visualize the class outputs as images, we need to get the images. The images should be passed in the same order as the output layer of the network. Important! <br>
- If you want to visualize power measurements during streaming inference, set `add_power_monitor_plot`=`True`.
- If you want to visualize readout images as class predictions during streaming you need to pass `add_readout_plot`=`True`.
- In order to show a prediction for each `N` milliseconds, set the parameter `spike_collection_interval`=`N`.
- In order to show the images, the paths of these images should be passed to `readout_images` parameter.
- In order to show a prediction only if there are more than a `threshold` number of events from that output, set the `readout_prediction_threshold`=`threshold`.
- In order to default to a certain class (in `DVSGesture` example, we use the last `other` class) set `readout_default_return_value`=`class_idx` (int).
- In order to limit the prediction between some thresholds (i.e. it is meaningless to make a prediction with too low and too high values) set `readout_default_threshold_low` and `readout_default_threshold_high` parameters.
- In all the chips except for `DynapcnnDevkit` you can monitor the power consumption in 5 different rows. These rows are `io`, `logic`, `ram`, `pixel-digital` and `pixel-analog`.
    - `io`: Power consumption of IO unit on the chip.
    - `logic`: Power consumption caused by operations in the chip and communication with spikes.
    - `ram`: Power consumption of `memory` used to store `states` and `weights`.
    - `pixel-digital`: Power consumption of the digital part (i.e. communication via AER) of pixels.
    - `pixel-analog`: Power consumtpion of the analog part. (i.e. pixels.
    ). 
- `DynapcnnDevkit` does not come with an on-board Dvs sensor, thus does not support the last two. If you want only the first 3 columns, pass `power_monitor_number_of_items`=`3`. If you want columns pass `5`.
- If you follow the naming conventions for readout images (i.e. `{labelidx}_{labelname}.png`) the feature names are going to be parsed from the image names. However, you can also pass labels manually using `feature_names`=`["class01", "class02", ...]`. That can be used also while labeling `SpikeCountPlot` legend, when `ReadoutLayer` is not preferred.
- The last layer feature count is going to be automatically extracted from the model. However you can pass it manually `feature_count`=`count`. This can be useful for plotting purposes, when there are other classes that you did not take into account in the model you trained.
- `extra_arguments`: You can pass the function names and variables for the individual plots for `spike_count`, `readout` and `power_measurement` plots. Available function names and argument types can be found in the following link: [here](https://synsense-sys-int.gitlab.io/samna/reference/viz/imgui/index.html#samna-viz-imgui)

```
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer
visualizer = DynapcnnVisualizer(
    window_scale=(4, 8),
    dvs_shape=(128, 128),
    add_power_monitor_plot=True,
    add_readout_plot=True,
    spike_collection_interval=500,
    readout_images=sorted([os.path.join(icons_folder_path, f) for f in os.listdir(icons_folder_path)])
)
```

#### Finally connect your model to the visualizer
Please note that `DynapcnnVisualizer` class is powered by Just-In-Time (JIT) compilation in C++. If you are running this on a computer, which does not have a powerful CPU this may take a while. You will see the window spawning, but you will not see anything displayed on it until the compilation is complete.

```
visualizer.connect(hardware_compatible_model)
```

### Try it yourself
The example script that runs the visualizer can be found under `/examples/visualizer/gesture_viz.py` .


#### MacOS users
Due to the difference in the behaviour of python's multiprocessing library on MacOS, you should run the `examples/visualizer/gesture_viz.py` script with `-i` flag. `python -i /examples/visualizer/gesture_viz.py` .