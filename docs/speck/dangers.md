Dangers
=======

Biases/Leak are not encouraged
-------------------------

We suggest to stay away from the use of biases, especially on layers with a large number of neurons. 
They introduce a lot of overhead on the chip as it is not a sparse operation and therefore increase the amount of computation and consequently power.

However, if you still like to try the leak/bias, please refer to ["how to leak neuron"](../getting_started/notebooks/leak_neuron.ipynb)


Event driven vs time-step based evaluation
------------------------------------------

These are not the same thing, even though *ALL* of our pytorch model simulations are time-step based!
So you might run into cases where your hardware produces a different result to that from your simulation.


Spike generation function
-------------------------
Sinabs supports multiple spike generation functions such as *SingleSpike*, *MultiSpike*, etc. We have observed empirically, that the best match between Sinabs simulation and on-chip inference is achieved with the *MultiSpike* method for common network architectures. Therefore, we suggest to use this method if the end goal is to run the trained model on one of our chips. 