# Chip Errata(Hardware Bugs)

## Speck2e/2f

 - Channel index mapping error between the output DYNAP-CNN core/layer and readout layer: 

   - channel 0 in DYNAP-CNN core mapped to channel 1 in readout layer.
   - channel 4 in DYNAP-CNN core mapped to channel 2 in readout layer.
   - channel 8 in DYNAP-CNN core mapped to channel 3 in readout layer.
   - ...
   
So if you wanted to use the readout layer of the devkit, please make sure your last convolutional layers'
weights is correctly mapped. In `sinabs-dynapcnn` we provide a utility function:

`sinabs.backend.dynapcnn.utils.extend_readout_layer`

to help you re-mapping the last convolutional layer's weights, for more details see 
["How to Use Readout Layer"](../getting_started/notebooks/using_readout_layer.ipynb)