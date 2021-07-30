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

