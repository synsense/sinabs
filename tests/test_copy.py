import torch
from copy import deepcopy


def test_deepcopy_iaf():
    from sinabs.layers import IAF, IAFSqueeze

    kwargs = dict(threshold=0.5, threshold_low=-0.4, membrane_subtract=0.45, window=0.8)

    for reset in (True, False):

        kwargs["membrane_reset"] = reset

        layer = IAF(**kwargs)

        layer_squeeze_nts = IAFSqueeze(**kwargs, num_timesteps=10)

        layer_squeeze_batch = IAFSqueeze(**kwargs, batch_size=10)

        for layer_orig in (layer, layer_squeeze_batch, layer_squeeze_nts):

            # Modify default parameters and buffers
            for b in layer_orig.buffers():
                b += 1
            for p in layer_orig.parameters():
                p += 1

            layer_copy = deepcopy(layer_orig)

            for p0, p1 in zip(layer_orig.parameters(), layer_copy.parameters()):
                assert (p0 == p1).all()
                assert p0 is not p1
            for b0, b1 in zip(layer_orig.buffers(), layer_copy.buffers()):
                assert (b0 == b1).all()
                assert b0 is not b1

            assert layer_copy.threshold == layer_orig.threshold
            assert layer_copy.threshold_low == layer_orig.threshold_low
            assert layer_copy.membrane_subtract == layer_orig.membrane_subtract
            assert layer_copy.membrane_reset == layer_orig.membrane_reset
            assert layer_copy.learning_window == layer_orig.learning_window
            if hasattr(layer_orig, "batch_size"):
                assert layer_orig.batch_size == layer_copy.batch_size
            if hasattr(layer_orig, "num_timesteps"):
                assert layer_orig.num_timesteps == layer_copy.num_timesteps


def test_deepcopy_lif():
    from sinabs.layers import LIF, LIFSqueeze, LIFRecurrent

    for train_alphas in (True, False):
        kwargs = dict(tau_mem=torch.tensor(30.), tau_syn=torch.tensor(10.))

        input_current = torch.rand(10,10,10)
        kwargs["train_alphas"] = train_alphas

        layer = LIF(**kwargs)
        layer_squeeze_nts = LIFSqueeze(**kwargs, num_timesteps=10)
        layer_squeeze_batch = LIFSqueeze(**kwargs, batch_size=10)
        layer_recurrent = LIFRecurrent(**kwargs, rec_connectivity=torch.nn.Linear(10,10))

        for layer_orig in (layer, layer_squeeze_batch, layer_squeeze_nts, layer_recurrent):
            layer_orig(input_current)

#             breakpoint()
            layer_copy = deepcopy(layer_orig)
            
            for p0, p1 in zip(layer_orig.parameters(), layer_copy.parameters()):
                assert (p0 == p1).all()
                assert p0 is not p1
            for b0, b1 in zip(layer_orig.buffers(), layer_copy.buffers()):
                assert (b0 == b1).all()
                assert b0 is not b1

            if train_alphas:
                assert layer_copy.alpha_mem == layer_orig.alpha_mem
            else:
                assert layer_copy.tau_mem == layer_orig.tau_mem
            assert layer_copy.activation_fn == layer_orig.activation_fn
            assert layer_copy.threshold_low == layer_orig.threshold_low
            assert layer_copy.train_alphas == layer_orig.train_alphas
            if hasattr(layer_orig, "batch_size"):
                assert layer_orig.batch_size == layer_copy.batch_size
            if hasattr(layer_orig, "num_timesteps"):
                assert layer_orig.num_timesteps == layer_copy.num_timesteps
            if hasattr(layer_orig, "rec_connectivity"):
                assert layer_orig.rec_connectivity.in_features == layer_copy.rec_connectivity.in_features


def test_deepcopy_alif():
    from sinabs.layers import ALIF, ALIFSqueeze

    kwargs = dict(
        tau_mem=20,
        tau_adapt=10,
        adapt_scale=1.3,
        threshold=0.5,
        threshold_low=-0.4,
        membrane_subtract=0.45,
    )

    for reset in (True, False):

        kwargs["membrane_reset"] = reset

        layer = ALIF(**kwargs)

        layer_squeeze_nts = ALIFSqueeze(**kwargs, num_timesteps=10)

        layer_squeeze_batch = ALIFSqueeze(**kwargs, batch_size=10)

        for layer_orig in (layer, layer_squeeze_batch, layer_squeeze_nts):

            # Modify default parameters and buffers
            for b in layer_orig.buffers():
                b += 1
            for p in layer_orig.parameters():
                p += 1

            layer_copy = deepcopy(layer_orig)

            for p0, p1 in zip(layer_orig.parameters(), layer_copy.parameters()):
                assert (p0 == p1).all()
                assert p0 is not p1
            for (name, b0), (name, b1) in zip(layer_orig.named_buffers(), layer_copy.named_buffers()):
                assert (b0 == b1).all()
                assert b0 is not b1

            assert layer_copy.threshold == layer_orig.threshold
            assert layer_copy.threshold_low == layer_orig.threshold_low
            assert layer_copy.membrane_subtract == layer_orig.membrane_subtract
            assert layer_copy.membrane_reset == layer_orig.membrane_reset
            assert layer_copy.tau_mem == layer_orig.tau_mem
            assert layer_copy.tau_adapt == layer_orig.tau_adapt
            assert layer_copy.adapt_scale == layer_orig.adapt_scale
            if hasattr(layer_orig, "batch_size"):
                assert layer_orig.batch_size == layer_copy.batch_size
            if hasattr(layer_orig, "num_timesteps"):
                assert layer_orig.num_timesteps == layer_copy.num_timesteps


def test_deepcopy_expleak():
    from sinabs.layers import ExpLeak, ExpLeakSqueeze

    layer = ExpLeak(tau_leak=10)
    layer_squeeze_nts = ExpLeakSqueeze(tau_leak=10, num_timesteps=10)
    layer_squeeze_batch = ExpLeakSqueeze(tau_leak=10, batch_size=10)

    for layer_orig in (layer, layer_squeeze_batch, layer_squeeze_nts):

        # Modify default parameters and buffers
        for b in layer_orig.buffers():
            b += 1
        for p in layer_orig.parameters():
            p += 1

        layer_copy = deepcopy(layer_orig)

        for p0, p1 in zip(layer_orig.parameters(), layer_copy.parameters()):
            assert (p0 == p1).all()
            assert p0 is not p1
        for b0, b1 in zip(layer_orig.buffers(), layer_copy.buffers()):
            assert (b0 == b1).all()
            assert b0 is not b1

        assert layer_copy.tau_leak == layer_orig.tau_leak
        if hasattr(layer_orig, "batch_size"):
            assert layer_orig.batch_size == layer_copy.batch_size
        if hasattr(layer_orig, "num_timesteps"):
            assert layer_orig.num_timesteps == layer_copy.num_timesteps
