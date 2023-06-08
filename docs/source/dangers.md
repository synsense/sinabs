Dangers
=======

Biases/Leak are not encouraged
-------------------------

We suggest to stay away from the use of biases, especially on layers with a large number of neurons. 
They introduce a lot of overhead on the chip as it is not a sparse operation and therefore increase the amount of computation and consequently power.

Event driven vs time-step based evaluation
------------------------------------------

These are not the same thing, even though *ALL* of our pytorch model simulations are time-step based!
So you might run into cases where your hardware produces a different result to that from your simulation.

