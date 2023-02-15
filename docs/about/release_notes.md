# Release notes

## v1.2.0 (15/02/2023)
Included the SNNAnalyzer module to collect model statistics such as number of synops or neurons automatically.

## v1.0.1 (26/08/2022)
Mostly making v0.3.x stable, with a revamped documentation that includes a gallery and how-tos!

## v0.3.0
This is a major overhaul which rewrites a large part of the package. 

* Addition of leaky models such as Leaky Integrate and Fire (LIF), Exponential Leaky (ExpLeak) and Adaptive LIF (ALIF).
* Activation module: from sinabs.activation you'll now be able to pick and choose different spike generation, reset mechanism and surrogate gradient functions. You can pass them to the neuron model (LIF, IAF, ...) of your liking if you want to alter the default behavior.
* new documentation on readthedocs
* SpikingLayer has been renamed to IAF (Integrate and Fire).
* State variable names changed and new ones have been added: 'state' is now called 'v_mem' and 'i_syn' is added for neuron layers that use tau_syn.
* New neuron features: support for recurrent connections, recording internal states, normalising inputs by taus, initialisation with shape and more.
* We moved our repo to Github and changed the CI pipeline to Github actions.

## v0.2.1
- TorchLayer renamed to Layer
    - Added a depricated class TorchLayer with warning
- Added some new layer types
    - BPTT enabled spike activation layer
    - SpikingTemporalConv1dLayer
