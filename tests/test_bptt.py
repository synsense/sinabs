def test_iaf_inference_vs_bptt():
    import sinabs.layers as sl
    import torch
    from sinabs.layers.iaf_bptt import SpikingLayer

    n_inp = 100

    # define input
    inp = torch.rand((50, n_inp))

    sl_bptt = SpikingLayer(input_shape=(n_inp,), threshold=1.0, threshold_low=-1)

    class SpikingPassthrough(sl.SpikingLayer):
        def __init__(self, input_shape, threshold, threshold_low):
            super().__init__(
                input_shape=input_shape,
                threshold=threshold,
                threshold_low=threshold_low,
                membrane_subtract=threshold,
                membrane_reset=0,
                layer_name="passthrough",
                negative_spikes=False,
            )

        def get_output_shape(self, input_shape):
            return input_shape

        def synaptic_output(self, input_spikes: torch.Tensor) -> torch.Tensor:
            return input_spikes

    sl = SpikingPassthrough(input_shape=(n_inp,), threshold=1, threshold_low=-1)
