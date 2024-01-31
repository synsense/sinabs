def test_specksim_conv_layer():
    import numpy as np
    import samna

    weight_shape = (2, 2, 3, 3)
    event = samna.specksim.events.Spike(0, 2, 2, 10)  # channel  # y  # x  # timestamp
    events = [event]

    conv_weights = np.ones(shape=weight_shape).tolist()

    conv_layer = samna.specksim.Convolution2d(
        weight_shape[1],  # in channels
        weight_shape[0],  # out channels
        (weight_shape[2], weight_shape[3]),  # kernel size (y, x)
        (5, 5),  # input shape (y, x)
        (1, 1),  # stride (y, x)
        (0, 0),  # padding (y, x)
    )
    conv_layer.set_weights(conv_weights)

    out = conv_layer.forward(events)
    assert len(out) == weight_shape[1] * weight_shape[2] * weight_shape[3]


def test_specksim_conv_filter():
    import time

    import numpy as np
    import samna

    weight_shape = (2, 2, 3, 3)
    event = samna.specksim.events.Spike(0, 2, 2, 10)  # channel  # y  # x  # timestamp
    events = [event]
    conv_weights = np.ones(shape=weight_shape).tolist()

    graph = samna.graph.EventFilterGraph()
    source, conv_filter, sink = graph.sequential(
        [
            samna.BasicSourceNode_specksim_events_spike(),
            samna.specksim.nodes.SpecksimConvolutionalFilterNode(),
            samna.BasicSinkNode_specksim_events_weighted_event(),
        ]
    )
    conv_filter.set_parameters(
        weight_shape[1],  # in channels
        weight_shape[0],  # out channels
        (weight_shape[2], weight_shape[3]),  # kernel shape y, x
        (5, 5),  # input shape y, x
        (1, 1),  # stride
        (0, 0),  # padding
    )
    conv_filter.set_weights(conv_weights)
    graph.start()

    source.write(events)
    time.sleep(1)
    out = sink.get_events()
    assert len(out) == weight_shape[1] * weight_shape[2] * weight_shape[3]
    graph.stop()


def test_specksim_iaf_layer():
    import numpy as np
    import samna

    events = [
        samna.specksim.events.WeightedEvent(
            0, 2, 2, 10, 0.2  # channel  # y  # x  # timestamp  # weight
        )
    ] * 6

    iaf_layer = samna.specksim.IntegrateAndFire(
        2,  # in channels
        (5, 5),  # input shape y, x
        1.0,  # spike threshold
        0.0,  # min v mem
    )
    out = iaf_layer.forward(events)
    assert len(out) == 1

    # test resetting the states
    states_after_inference = np.array(iaf_layer.get_v_mem())  # sum: 0.2
    iaf_layer.reset_states()
    states_after_reset = np.array(iaf_layer.get_v_mem())  # sum: 0.0
    assert states_after_reset.sum() < states_after_inference.sum()


def test_specksim_iaf_filter():
    import time

    import samna

    events = [
        samna.specksim.events.WeightedEvent(
            0, 2, 2, 10, 0.2  # channel  # y  # x  # timestamp  # weight
        )
    ] * 5

    graph = samna.graph.EventFilterGraph()

    source, iaf_node, sink = graph.sequential(
        [
            samna.BasicSourceNode_specksim_events_weighted_event(),
            samna.specksim.nodes.SpecksimIAFFilterNode(),
            samna.BasicSinkNode_specksim_events_spike(),
        ]
    )
    iaf_node.set_parameters(
        2,  # in channels
        (5, 5),  # input shape y, x
        1.0,  # spike threshold
        0.0,  # min v mem
    )
    graph.start()
    source.write(events)
    time.sleep(2)
    out = sink.get_events()
    graph.stop()
    assert len(out) == 1


def test_specksim_sum_pooling_layer():
    import samna

    y, x = 4, 4
    pool_y, pool_x = 2, 2
    pool_layer = samna.specksim.SumPooling((pool_y, pool_x), (8, 8))
    event = samna.specksim.events.Spike(0, y, x, 10)  # channel  # y  # x  # timestamp
    events = [event]
    out = pool_layer.forward(events)
    out_event = out[0]
    assert out_event.x == x // pool_x
    assert out_event.y == y // pool_y


def test_specksim_sum_pooling_filter():
    import time

    import samna

    y, x = 4, 4
    pool_y, pool_x = 2, 2
    event = samna.specksim.events.Spike(0, y, x, 10)  # channel  # y  # x  # timestamp
    events = [event]
    graph = samna.graph.EventFilterGraph()
    source, pooling_node, sink = graph.sequential(
        [
            samna.BasicSourceNode_specksim_events_spike(),
            samna.specksim.nodes.SpecksimSumPoolingFilterNode(),
            samna.BasicSinkNode_specksim_events_spike(),
        ]
    )
    pooling_node.set_parameters((pool_y, pool_x), (8, 8))

    graph.start()
    source.write(events)
    time.sleep(1)
    out = sink.get_events()
    graph.stop()
    out_event = out[0]
    assert out_event.y == y // pool_y
    assert out_event.x == x // pool_x
