from copy import deepcopy

import torch


def test_deepcopy_iaf():
    from sinabs.layers import IAF, IAFSqueeze

    input_current = torch.rand(10, 10, 10)
    kwargs = dict(min_v_mem=-0.4)

    layer = IAF(**kwargs)
    layer_squeeze_nts = IAFSqueeze(**kwargs, num_timesteps=10)
    layer_squeeze_batch = IAFSqueeze(**kwargs, batch_size=10)

    for layer_orig in (layer, layer_squeeze_batch, layer_squeeze_nts):
        layer_orig(input_current)

        layer_copy = deepcopy(layer_orig)

        for p0, p1 in zip(layer_orig.parameters(), layer_copy.parameters()):
            assert (p0 == p1).all()
            assert p0 is not p1
        for b0, b1 in zip(layer_orig.buffers(), layer_copy.buffers()):
            assert (b0 == b1).all()
            assert b0 is not b1

        assert layer_copy.spike_threshold == layer_orig.spike_threshold
        assert layer_copy.min_v_mem == layer_orig.min_v_mem
        if hasattr(layer_orig, "batch_size"):
            assert layer_orig.batch_size == layer_copy.batch_size
        if hasattr(layer_orig, "num_timesteps"):
            assert layer_orig.num_timesteps == layer_copy.num_timesteps


def test_deepcopy_iaf_uninitialized():
    from sinabs.layers import IAF, IAFSqueeze

    kwargs = dict(min_v_mem=-0.4)

    layer = IAF(**kwargs)
    layer_squeeze_nts = IAFSqueeze(**kwargs, num_timesteps=10)
    layer_squeeze_batch = IAFSqueeze(**kwargs, batch_size=10)

    for layer_orig in (layer, layer_squeeze_batch, layer_squeeze_nts):
        layer_copy = deepcopy(layer_orig)

        for p0, p1 in zip(layer_orig.parameters(), layer_copy.parameters()):
            assert (p0 == p1).all()
            assert p0 is not p1
        for b0, b1 in zip(layer_orig.buffers(), layer_copy.buffers()):
            assert (b0 == b1).all()
            assert b0 is not b1

        assert layer_copy.spike_threshold == layer_orig.spike_threshold
        assert layer_copy.min_v_mem == layer_orig.min_v_mem
        if hasattr(layer_orig, "batch_size"):
            assert layer_orig.batch_size == layer_copy.batch_size
        if hasattr(layer_orig, "num_timesteps"):
            assert layer_orig.num_timesteps == layer_copy.num_timesteps


def test_deepcopy_lif():
    from sinabs.layers import LIF, LIFSqueeze

    for train_alphas in (True, False):

        input_current = torch.rand(10, 10, 10)
        kwargs = dict(tau_mem=torch.tensor(30.0), tau_syn=torch.tensor(10.0))

        kwargs["train_alphas"] = train_alphas

        layer = LIF(**kwargs)
        layer_squeeze_batch = LIFSqueeze(**kwargs, batch_size=10)
        layer_squeeze_nts = LIFSqueeze(**kwargs, num_timesteps=10)
        #         layer_recurrent = LIFRecurrent(**kwargs, rec_connect=torch.nn.Linear(10,10))

        for layer_orig in (layer, layer_squeeze_batch, layer_squeeze_nts):
            layer_orig(input_current)

            layer_copy = deepcopy(layer_orig)

            for p0, p1 in zip(layer_orig.parameters(), layer_copy.parameters()):
                assert (p0 == p1).all()
                assert p0 is not p1
            for b0, b1 in zip(layer_orig.buffers(), layer_copy.buffers()):
                assert (b0 == b1).all()
                assert b0 is not b1

            assert layer_copy.spike_threshold == layer_orig.spike_threshold
            if train_alphas:
                assert layer_copy.alpha_mem == layer_orig.alpha_mem
            else:
                assert layer_copy.tau_mem == layer_orig.tau_mem
            assert layer_copy.spike_fn == layer_orig.spike_fn
            assert layer_copy.min_v_mem == layer_orig.min_v_mem
            assert layer_copy.train_alphas == layer_orig.train_alphas
            if hasattr(layer_orig, "batch_size"):
                assert layer_orig.batch_size == layer_copy.batch_size
            if hasattr(layer_orig, "num_timesteps"):
                assert layer_orig.num_timesteps == layer_copy.num_timesteps
            if hasattr(layer_orig, "rec_connect"):
                assert (
                    layer_orig.rec_connect.in_features
                    == layer_copy.rec_connect.in_features
                )


def test_deepcopy_lif_uninitialized():
    from sinabs.layers import LIF, LIFSqueeze

    for train_alphas in (True, False):
        kwargs = dict(tau_mem=torch.tensor(30.0), tau_syn=torch.tensor(10.0))

        kwargs["train_alphas"] = train_alphas

        layer = LIF(**kwargs)
        layer_squeeze_batch = LIFSqueeze(**kwargs, batch_size=10)
        layer_squeeze_nts = LIFSqueeze(**kwargs, num_timesteps=10)
        #         layer_recurrent = LIFRecurrent(**kwargs, rec_connect=torch.nn.Linear(10,10))

        for layer_orig in (layer, layer_squeeze_batch, layer_squeeze_nts):

            layer_copy = deepcopy(layer_orig)

            for p0, p1 in zip(layer_orig.parameters(), layer_copy.parameters()):
                assert (p0 == p1).all()
                assert p0 is not p1
            for b0, b1 in zip(layer_orig.buffers(), layer_copy.buffers()):
                assert (b0 == b1).all()
                assert b0 is not b1

            assert layer_copy.spike_threshold == layer_orig.spike_threshold
            if train_alphas:
                assert layer_copy.alpha_mem == layer_orig.alpha_mem
            else:
                assert layer_copy.tau_mem == layer_orig.tau_mem
            assert layer_copy.spike_fn == layer_orig.spike_fn
            assert layer_copy.min_v_mem == layer_orig.min_v_mem
            assert layer_copy.train_alphas == layer_orig.train_alphas
            if hasattr(layer_orig, "batch_size"):
                assert layer_orig.batch_size == layer_copy.batch_size
            if hasattr(layer_orig, "num_timesteps"):
                assert layer_orig.num_timesteps == layer_copy.num_timesteps
            if hasattr(layer_orig, "rec_connect"):
                assert (
                    layer_orig.rec_connect.in_features
                    == layer_copy.rec_connect.in_features
                )


def test_deepcopy_alif():
    from sinabs.layers import ALIF

    kwargs = dict(
        tau_mem=20.0,
        tau_adapt=10.0,
        adapt_scale=1.3,
        min_v_mem=-0.4,
    )

    layer_orig = ALIF(**kwargs)

    layer_orig(torch.rand(10, 10, 10))

    layer_copy = deepcopy(layer_orig)

    for p0, p1 in zip(layer_orig.parameters(), layer_copy.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for (name, b0), (name, b1) in zip(
        layer_orig.named_buffers(), layer_copy.named_buffers()
    ):
        assert (b0 == b1).all()
        assert b0 is not b1

    assert layer_copy.min_v_mem == layer_orig.min_v_mem
    assert layer_copy.spike_fn == layer_orig.spike_fn
    assert layer_copy.tau_mem == layer_orig.tau_mem
    assert layer_copy.tau_adapt == layer_orig.tau_adapt
    assert layer_copy.adapt_scale == layer_orig.adapt_scale
    if hasattr(layer_orig, "batch_size"):
        assert layer_orig.batch_size == layer_copy.batch_size
    if hasattr(layer_orig, "num_timesteps"):
        assert layer_orig.num_timesteps == layer_copy.num_timesteps


def test_deepcopy_expleak():
    from sinabs.layers import ExpLeak, ExpLeakSqueeze

    layer = ExpLeak(tau_mem=10.0)
    layer_squeeze_nts = ExpLeakSqueeze(tau_mem=10.0, num_timesteps=10)
    layer_squeeze_batch = ExpLeakSqueeze(tau_mem=10.0, batch_size=10)

    for layer_orig in (layer, layer_squeeze_batch, layer_squeeze_nts):
        layer_orig(torch.rand(10, 10, 10))

        layer_copy = deepcopy(layer_orig)

        for p0, p1 in zip(layer_orig.parameters(), layer_copy.parameters()):
            assert (p0 == p1).all()
            assert p0 is not p1
        for b0, b1 in zip(layer_orig.buffers(), layer_copy.buffers()):
            assert (b0 == b1).all()
            assert b0 is not b1

        assert layer_copy.tau_mem == layer_orig.tau_mem
        if hasattr(layer_orig, "batch_size"):
            assert layer_orig.batch_size == layer_copy.batch_size
        if hasattr(layer_orig, "num_timesteps"):
            assert layer_orig.num_timesteps == layer_copy.num_timesteps
