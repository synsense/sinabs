activation
==========
.. currentmodule:: sinabs.activation

Spiking layers can choose any combination of spike generation, reset mechanism and surrogate gradient function.

Spike Generation
------------------
.. autosummary::
    :toctree: generated/
    :template: class_activation.rst

    SingleSpike
    MultiSpike
    MaxSpike

Reset Mechanisms
----------------
.. autosummary::
    :toctree: generated/
    :template: class_activation.rst

    MembraneReset
    MembraneSubtract

Surrogate Gradient Functions
----------------------------
.. autosummary::
    :toctree: generated/
    :template: class_activation.rst

    SingleExponential
    PeriodicExponential
    Heaviside
    Gaussian
    MultiGaussian
