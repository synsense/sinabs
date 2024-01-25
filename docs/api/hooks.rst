hooks
==========

This module provides hooks that can be registered with layers and modules to collect statistics during a forward pass.


Hooks
-----
.. autofunction:: sinabs.hooks.firing_rate_hook
.. autofunction:: sinabs.hooks.firing_rate_per_neuron_hook
.. autofunction:: sinabs.hooks.input_diff_hook
.. autofunction:: sinabs.hooks.conv_layer_synops_hook
.. autofunction:: sinabs.hooks.linear_layer_synops_hook
.. autoclass:: sinabs.hooks.ModelSynopsHook
    :members:


Helper functions
----------------
.. autofunction:: sinabs.hooks.register_synops_hooks
.. autofunction:: sinabs.hooks.get_hook_data_dict
.. autofunction:: sinabs.hooks.conv_connection_map
.. autofunction:: sinabs.hooks._extract_single_input

