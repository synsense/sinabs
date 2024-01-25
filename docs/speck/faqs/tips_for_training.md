# Troubleshooting and Tips

To train an SNN which aimed to deployed to Speck, we share the following tips for the users. Hope to make the deployment
process smoother.

## Memory Constrains On The Hardware

In the ["overview of devkit"](/getting_started/overview.md) we mentioned that each core on our devkit has its own memory
constrains:

![memoryconstrains](/_static/Overview/memory_constraints.png)

If you want your SNN can be deployed to the devkit, please ensure your network has an appropriate architecture.
The way to calculate the neuron memory and kernel memory can be found [here.](/getting_started/overview.md)

## SynOps/s Bandwidth Constrains On The Hardware

### What Is SynOps/s Bandwidth?
A Synaptic-Operation (SynOp) can be defined as:

All the operations involved in the lifecycle of a spike arriving at a layer until it updates the neuron’s
membrane potential(and generates a spike if appropriate). Operations in the lifecycle of the arrived spike
consist of:

`
 "Logic" −> "Kernel Memory Read" −> "Logic" −> "Neuron Read" −> "Neuron Write" −> "Logic" 
`

Each core/layer on the DynapCNN/Speck has an upper limit on the number of synaptic-operations it can execute per second.
This upper limit is the "bandwidth of SynOps/s". 

If the number of SynOps/s for a single core exceeds the limit, the entire chip stalls, and the
behaviour of the chip is unpredictable under that circumstance. So during designing and training
stage of your SNN, you should keep the SynOps/s bandwidth in mind and don't let your SNN exceeds
the bandwidth.

### How To Spot A Model Exceeding The BandWidth?
There is currently no way to definitively diagnose if a chip is exceeding its bandwidth,
except with an oscilloscope. However, if the bandwidth of the chip is exceeded,
it is often accompanied by the following phenomena: 

1. A significant delay on the output is observed when you manually write pre-recorded input events
   to the devkit.
2. The DVS events visualized by samnagui show **striped edge pattern**, 
   and a significant delay can also be observed for real-time input DVS events.

![exceeding_bandwidth](/_static/tips_for_training/exceeding_bandwidth.png)

### How To Estimate The SynOps/s For A Model?
We have the following formula for estimating the 
* {math}`N_{spk}/s`, stands for the input number of spikes of one DynapCNN core per second.
* {math}`C_{out}`, stands for the number of the output channels of the convolutional layer.
* {math}`k_x` and {math}`k_y` stands for the convolutional kernel size.
* {math}`s_x` and {math}`s_y` stands for the tiling stride of the convolutional kernel.

Then the estimated SynOps/s of the DynapCNN core is:

* {math}`SynOps/s \approx \frac{N_{spk}/s \times C_{out} \times k_x \times k_y}{s_x \times s_y}`

In [sinabs](https://github.com/synsense/sinabs), an auxiliary class named `SNNAnalyzer` is provided for users
to estimate the SynOps/s. By just a few lines of code, you can get the SynOps/s of each layer of your model.

```python
from sinabs.synopcounter import SNNAnalyzer

# dt refers to the time interval (micro-second) of a single time step of your spike train tensor
analyzer = SNNAnalyzer(my_snn, dt=my_raster_data_dt)
output = my_snn(my_raster_data)  # forward pass
layer_stats = analyzer.get_layer_statistics()

for layer_name, _ in my_snn.named_modules():
    synops_per_sec = layer_stats["parameter"][layer_name]["synops/s"]
    print(f"SynOps/s of layer {layer_name} is: {synops_per_sec}")
```

### How To Prevent From Exceeding SynOps/s Bandwidth？
The following tricks will help the developer to reduce the SynOps for their SNN model.

1. Use smaller convolutional kernel.
2. Use fewer numbers of convolutional kernels i.e. less output channels.
3. Add "SynOps Loss" as a regularisation item during the training stage. A bare minimum example
   demonstrate the "SynOps Loss" is provided below:
```python
import torch
from sinabs.synopcounter import SNNAnalyzer

# use SNNAnalyzer to obtain SynOps/s
analyzer = SNNAnalyzer(my_snn, dt=my_raster_data_dt)
output = my_snn(my_raster_data)  # forward pass
layer_statistics = analyzer.get_layer_statistics()["parameter"]
synops_of_every_layer = [
    stats["synops/s"] for layer, stats in layer_statistics.items()
]

# suppose my_snn is a 4-layer SNN
# manually set the upper limit for each layer
synops_upper_limit = [1e5, 2e5, 2e5, 1e4]

# calculate synops loss
synops_loss = 0
for synops_per_layer, limit in zip(synops_of_every_layer, synops_upper_limit):
    # punish only if the synops_per_layer higher than the limit
    residual = (synops_per_layer - limit) / limit
    synops_loss += torch.nn.functional.relu(residual)
```
4. Switch on the 
   ["decimator"](https://synsense-sys-int.gitlab.io/samna/reference/speck2e/configuration/index.html?highlight=decimation#samna.speck2e.configuration.CnnLayerConfig.output_decimator_enable)
   on your DynapCNN Core. In the devkit, each DynapCNN Core is equipped with a decimator block at its data path output.
   The decimator block enables the user to reduce the spike rate at the output of a DynapCNN Core. By default,
   the decimator is disabled, and the code below shows an example of enabling the decimator of the chip by modifying
   the samna configuration.
   
```python
from sinabs.backend.dynapcnn import DynapcnnNetwork

# generate the samna configuration
dynapcnn = DynapcnnNetwork(snn=YOUR_SNN, input_shape=(1, 128, 128), dvs_input=False)
# suppose your snn is a 3-layer model
samna_cfg = dynapcnn.make_config(device="speck2fmodule", chip_layers_ordering=[0, 1, 2])

# turn on the decimator on Core #0
layer_idx = 0
samna_cfg.cnn_layers[layer_idx].output_decimator_enable = True
# 1 spike passed for every 4 output spikes
# the number of output events from Core#0 will be reduced to 1/4 of the original
samna_cfg.cnn_layers[layer_idx].output_decimator_interval = 0b001
```

more details about the "decimator" can be found [here.](https://synsense-sys-int.gitlab.io/samna/reference/speck2f/configuration/index.html?highlight=output_decimator#samna.speck2f.configuration.CnnLayerConfig.output_decimator_interval)

## The Reset Mechanism Of Neuron Membrane Potential
Our devkit provides two types of reset mechanism for the spiking neuron's membrane potential.

1. Reset to 0 after firing 1 spike.
2. Subtract by spiking-threshold after firing 1 spike.

By default, our devkit use the second strategy for membrane potential reset.

If you use an ANN-to-SNN conversion, then you should choose the second one strategy for resetting membrane-potential.

If you train an SNN with a "reset to 0" strategy, then you should choose the first one strategy.

The `samna configuration` for each layer has a boolean attribute called 
["return_to_zero"](https://synsense-sys-int.gitlab.io/samna/reference/speck2f/configuration/index.html?highlight=return_to_zero#samna.speck2f.configuration.CnnLayerConfig.return_to_zero).
- If it is set to be `True`, then the hardware execute the first resetting strategy. 
- If it is set to be `False`, the hardware execute the second strategy.

```python
from sinabs.backend.dynapcnn import DynapcnnNetwork

# generate the samna configuration
dynapcnn = DynapcnnNetwork(snn=YOUR_SNN, input_shape=(1, 128, 128), dvs_input=False)
# suppose your snn is a 3-layer model
samna_cfg = dynapcnn.make_config(device="speck2fmodule", chip_layers_ordering=[0, 1, 2])

# suppose you choose the "reset_to_zero" mechanism
for layer in [0, 1, 2]:
    samna_cfg.cnn_layers[layer].return_to_zero = True
```

## Weight Clipping During Training

Usually, you should guarantee the maximum value of the convolutional layer weights be lower than the spiking threshold.
The **reason** is, the async spiking neuron design only supports the emission of a single spikes once 
the membrane potential goes above the spiking threshold. If the maximum value of the weights higher than the
spiking threshold, the postsynaptic IAF neuron on the chip might keep firing **as long as it keeps receiving input spikes.**
For example :

Suppose we have an IAF neuron `A`

* {math}`V_{A}`, stands for the membrane potential of the neuron `A`.
* {math}`\theta_{A}`, stands for the spiking threshold of the neuron `A`.
* For simplicity, we use 1 x 1 convolution kernel, 
  {math}`W_{A}` stands for the weight of the convolutional kernel, and {math}`W_{A}` > {math}` 2 \cdot \theta_{A}`.


1. Now neuron `A` receives an inputs spike, {math}`V_{A}` will be updated as {math}`V_{A} = V_{A} + W_{A}`. 
   Since {math}`W_{A}` > {math}` 2 \cdot \theta_{A}`, the updated {math}`V_{A}` now has grown higher than 
   {math}`2 \cdot \theta_{A}`.

2. Neuron `A` will only fire a single spike and {math}`V_{A}` will be updated as 
   {math}`V_{A} = V_{A} -\theta_{A}`(if we choose the "Subtract by spiking-threshold" resetting mechanism), 
   and the updated {math}`V_{A}` is still higher than the {math}`\theta_{A}`.

3. Neuron `A` **will remain silent although its {math}`V_{A}` > {math}`\theta_{A}` if no more input spike comes.**
   It only fires again only after it receiving a new input spike. Usually, the DVS sensor will keep sending input spikes
   into the DynapCNN Core and the neurons will keep firing forever, which is not friendly for low-power consumption 
   scenario.
   

## Selection Of Spike Function
You'd better choose "Multi-Spike" instead of "Single-Spike" due to the nature of our hardware as demonstrated
in the chapter above.


