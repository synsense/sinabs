# Overview

## DYNAP-CNN and SPECK

DYNAP-CNN/SPECK are a family of spiking neural network ASICs family, which is designed to focusing on convolution spiking neural network(SCNN) based vision processing tasks. DYNAP-CNN is a fully scalable, event-driven neuromorphic processor with up to 0.32M configurable spiking neurons and direct interface with external DVS. Speck™ is the world first neuromorphic device which integrates the DYNAP-CNN neuromorphic processor and a dynamic vision sensor(DVS) into a single SoC.

![image.png](/_static/Overview/speck_dynapcnn.png)


Currently, __sinabs-dynapcnn__ library provides the interface to serveral available versions of hardwares for DYNAPCNN/SPECK family:

| Device Name |Identifier |
| :----| :----: | 
| DYNAP-CNN Development Kit |*dynapcnndevkit | 
| Speck Tiny Development Kit|*speck2btiny|
| Speck Development Kit |*speck2edevkit|

__note__: * stands for the correspond software identifier we assigned to the different devkit

<br />

The general top level diagram of the DYNAP-CNN/SPECK chip is shown as follows:

![top_diagram](/_static/Overview/speck_top_level.png)


For DYNAP-CNN dev-kits, it allows the user to interact with the chip using external DVS sensors or pre-defined event data. Speck seriers devkit based on above, it additionally allows the user to use its embeded internal DVS for development.

Currently, Speck/Dynapcnn supports following External DVS sensor with the samna support:

* Inivation Davis346
* Inivation Davis240
* Inivation DVXplorer
* Prophesee EVK3-Gen 3.1VGA

# Backend: dynapcnn

To interact with these developmentkit, sinabs needs [samna](https://pypi.org/project/samna/) dependency to enables the chip configuration and network setting. As is shown in the figure below, dynapcnn backend provides a simple way to convert the network structure and parameters to the _SammnaConfiguration_ that can be used by samna to setup the chip. 


![sinab-dynapcnn](/_static/Overview/sinabs-dynapcnn-role.png)

# Chip Resources

For all DYNAP-CNN/SPECK family, they using the DYNAP asynchronous computing structure and have the similar computation resources. The detailed features are shown below

## Key Features

* Async input interface: Speck devkit supports configure both from the internal dvs sensor or external dvs resources

* Async output interface: 1: monitor bus output 2:readout layer output

* Interrupt: 4 ouput pins encodes max 15 + 1(no class) output

* 128x128 DVS array with every pixel can be configured to be killed

* 1x  Event(For DVS) pre-processing layer
* 9x  DYNAP-CNN Layer
* 1x  Readout layer that can does the prediction based on DYNAP-CNN layer output 

### Event Pre-processing Layer
The event pre-processing layer receives the input events coming from dvs or an externa source. Depending on the configuration, the layer can perform following operations based on events:

* Merge/Select the polarity of the input event stream
* Sum Pool with kerlnel size of 1x1, 2x2, 4x4
* ROI selection
* Mirroring the event stream
* Rotate 90 degrees

An Event noise filter is also included in the pre-processing layer, the filter can be enabled upon users setting. A general pre-processing pipeline for the preprocessing pipe line is shown as follow:

![image.png](/_static/Overview/event_preprocessing_pipeline.png)

### DYNAP-CNN Layer

The DYNAP-CNN Layer is the main hard ware representation of the designed spiking neural network structure. One of the main goal of the sinabs-dynapcnn is to provide an efficient, simply way to convert the `torch.Sequential` object to equivalent DYNAP-CNN layer configuration. Details of how to use the sinabs-dynapcnn interact with your designed SNNs, please visit tutorial



#### Feature
- Max input dimension for the layer: 128x128
- Max output feature map size: 64x64
- Max channel number: 1024
- Weight resolution 8bit
- Neuron State resolution 16bit
- Max convolutional kernel size 16x16
- Stride step:1,2,4,8
- Padding: [0,1,2,3,4,5,6,7]
- Sumpooling: 1x1, 2x2, 4x4
- Fanout:2

#### Internal Execution Order
A single chip consists of __9__ configurable computing cores(layers), each layer can be regarded as a combination of (Conv2d Operation --> Spiking Activation --> Sumpooling). These computation has to be configured in exact equivalent execution order. Each layer can be flexiblely configured to be communicate with other layers.


#### Async event-driven feature
information communication with layers are only in "Event based" format, the layer process the "incomming" event only at whenever a layer recieves it. Each layer can be configured to set 1-2 destination to the other layer. 


#### Memory Constraints and Network Sizing
Each layer has different memory constraints that split into kernel memory(for weight parameters) and neuron memory(spiking neuron states) as is shown below.  __Note:__ For the entire series of chips, dynapcnn/speck support precision of **8bit** int for kernel parameters and **16bit** int for neuron state precisions.

![memoryconstrains](/_static/Overview/memory_constraints.png)

<br />

with a convolutional layer is defined as

* {math}`c`, stands for input channel number
* {math}`f`, output channel number
* {math}`k_x` and {math}`k_y` kernel size

The theoretial number of entries required for kernel memory {math}`K_M` is then:

{math}`K_M = c \times f \times k_{x} \times k_{y}`

The actual number of entries required in chip because of the address encoding scheme, the actual total memory requires {math}`K_{MT}` is then:

{math}`K_{MT}=c \cdot 2^{log_{2}^{k_{x}k_{y}} + log_{2}^{f}}`

The required neuron memory entries depends on the output feature map where define the input feature map size {math}`c_{x}`, {math}`c_{y}`, stride {math}`s_{x}`, {math}`s_{y}`, padding {math}`p_{x}`, {math}`p_{y}`.

{math}`f_x = \frac{c_{x}-k_{x}+2p_{x}}{s_{x}} + 1`

{math}`f_y = \frac{c_{y}-k_{y}+2p_{y}}{s_{y}} + 1`

The actual neuron memory entries {math}`N_{M}` is then defined as:

{math}`N_{M} = f \times f_{x} \times f_{y}`


Taking an example of convolutional layer

    conv_layer = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
    
Assume the input dimension of 64x64 we could obtain the output feature map size as:

{math}`f_x = \frac{64-3+2 \times 1}{1} + 1 = 64`

{math}`f_y = \frac{64-3+2 \times 1}{1} + 1 = 64`


The actual kernel memory entries is calculated thus:

{math}`K_{MT}=16 \times 32 \times 4 \times 4 = 8Ki`

The actual Neuron memory entries is then:

{math}`N_{M} = 64 \times 64 \times 32 = 128Ki`

Where 128Ki neuron exceeds any available neuron memory contrains among 9 layers, thus this layer **CANNOT** be deployed on the chip

#### Leak operation

Each layer includes a leak generation block, which will update all the configured neuron states with provided leak values with a clock reference signal.**[tutorial on how to use leak feature]**

#### Congestion banlancer
In __latest devlopment kit(speck2e)__, each dynapcnn layer is equipped with a congestion banlancer upon its data path input. It is able to drops the incoming events at any time when the convolutional cores is overloaded in processing. 


#### Decimator
* In __latest development kit(speck2e)__, each dynapcnn layer is equipped with a decimator block at its output data path. The decimator enables the user reduce the spike rate at the output of a convolution layer. When enabled, the decimator allows 1 spikes to be passed every N output spikes(N=[2,4,8,16,32,128,256,512]).

### Readout Layer

The readout layer provides output data via the output serial interface. The time window where the readout window used is driven by an internal slow clock. The readout layer can be configured to have 1,16,32 times the provided clock cycle. 4 different addressing modes can be selected to assign input spikes to the readout layer to 15 available output classes.


### Internal Slow clock

**only avilable in latest speck2e**, a slow clock internally is used to support a number of time-cycle based features. 

* Leak clock, the DYNAPCNN layers including a leak operation that can operate on all the configured neuron states based on the clock setting.
* DVS pre-processing filter: The DVS filter uses the slow clock as the time reference to update its internal states
* Readout Layer: the readout layer uses the slow clock cycle as the moving average step to calculating the moving averages.

Note: The slow clock is internally generated by dividing the internal DVS raw event rates, which not always gurantee to be accurate.
