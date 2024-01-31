import pytest

from sinabs.backend.dynapcnn.chip_factory import ChipFactory

devices = ChipFactory.supported_devices


def generate_event_list(chip_factory):
    Spike = chip_factory.get_config_builder().get_samna_module().event.Spike

    # Note that timestamps are in us, chip factory dt in s
    return [
        Spike(timestamp=0, layer=2, x=1, y=0, feature=0),
        Spike(timestamp=9000, layer=2, x=3, y=3, feature=2),
        Spike(timestamp=9000, layer=2, x=3, y=3, feature=2),
        Spike(timestamp=14000, layer=0, x=1, y=2, feature=1),
        Spike(timestamp=14000, layer=2, x=0, y=4, feature=1),
        Spike(timestamp=19000, layer=2, x=2, y=1, feature=0),
    ]


@pytest.mark.parametrize("device", devices)
def test_events_to_raster(device):
    import torch

    factory = ChipFactory(device)
    events = generate_event_list(factory)

    dt = 1e-3
    raster = factory.events_to_raster(events, dt=dt)
    expected_raster = torch.zeros((20, 3, 5, 4))
    expected_raster[0, 0, 0, 1] = 1
    expected_raster[9, 2, 3, 3] = 2
    expected_raster[14, 1, 2, 1] = 1
    expected_raster[14, 1, 4, 0] = 1
    expected_raster[19, 0, 1, 2] = 1
    assert (raster == expected_raster).all()

    dt = 1e-2
    raster = factory.events_to_raster(events, dt=dt)
    expected_raster = torch.zeros((2, 3, 5, 4))
    expected_raster[0, 0, 0, 1] = 1
    expected_raster[0, 2, 3, 3] = 2
    expected_raster[1, 1, 2, 1] = 1
    expected_raster[1, 1, 4, 0] = 1
    expected_raster[1, 0, 1, 2] = 1
    assert (raster == expected_raster).all()

    shape = (4, 6, 6)
    dt = 1e-3
    raster = factory.events_to_raster(events, dt=dt, shape=shape)
    expected_raster = torch.zeros((20, *shape))
    expected_raster[0, 0, 0, 1] = 1
    expected_raster[9, 2, 3, 3] = 2
    expected_raster[14, 1, 2, 1] = 1
    expected_raster[14, 1, 4, 0] = 1
    expected_raster[19, 0, 1, 2] = 1
    assert (raster == expected_raster).all()

    dt = 1e-2
    raster = factory.events_to_raster(events, dt=dt, shape=shape)
    expected_raster = torch.zeros((2, *shape))
    expected_raster[0, 0, 0, 1] = 1
    expected_raster[0, 2, 3, 3] = 2
    expected_raster[1, 1, 2, 1] = 1
    expected_raster[1, 1, 4, 0] = 1
    expected_raster[1, 0, 1, 2] = 1
    assert (raster == expected_raster).all()


@pytest.mark.parametrize("device", devices)
def test_raster_to_events(device):
    import torch

    factory = ChipFactory(device)

    raster = (torch.rand((10, 2, 28, 28)) / 0.25).int()
    events = factory.raster_to_events(raster, layer=2)

    assert len(events) == raster.sum()
