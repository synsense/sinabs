Dangers
=======

Biases are a NO-NO
------------------

Do not use biases. They deploy a lot of overhead on the chip in terms of computation and consequently power.
At the moment, we do not support deploying models with biases BUT *there are no warnings to tell you when you do use biases*.
So watch out!


Event driven vs time-step based evaluation
------------------------------------------

These are not the same thing, even though *ALL* of our pytorch model simulations are time-step based!
So you might run into cases where your hardware produces a different result to that from your simulation.


Spike generation function
-------------------------
Sinabs supports multiple spike generation functions such as *SingleSpike*, *MultiSpike*, etc. We have observed empirically, that the best match between Sinabs simulation and on-chip inference is achieved with the *MultiSpike* method for common network architectures. Therefore, we suggest to use this method if the end goal is to run the trained model on one of our chips. 