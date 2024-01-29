import pytest
import samna
from hw_utils import find_open_devices, is_any_samna_device_connected

from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer


def X_available() -> bool:
    from subprocess import PIPE, Popen

    p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
    p.communicate()
    return p.returncode == 0


@pytest.mark.skipif(
    True,
    reason="A window needs to pop. Needs UI. Makes sense to check this test manually",
)
def test_visualizer_initialization():
    dvs_shape = (128, 128)
    spike_collection_interval = 500
    visualizer_id = 3

    visualizer = DynapcnnVisualizer(
        dvs_shape=dvs_shape, spike_collection_interval=spike_collection_interval
    )
    visualizer.create_visualizer_process(
        f"tcp://0.0.0.0:{visualizer.samna_visualizer_port}"
    )


def get_demo_dynapcnn_network():
    import torch.nn as nn

    import sinabs
    from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork

    ann = nn.Sequential(nn.Conv2d(2, 8, (3, 3)), nn.ReLU(), nn.AvgPool2d((2, 2)))
    snn = sinabs.from_model(ann, input_shape=(2, 64, 64), batch_size=1)

    dynapcnn_network = DynapcnnCompatibleNetwork(
        snn=snn, input_shape=(2, 64, 64), dvs_input=True
    )
    return dynapcnn_network


@pytest.mark.skipif(True, reason="No samna device found!")
def test_jit_compilation():
    dvs_shape = (128, 128)
    spike_collection_interval = 500
    visualizer_id = 3

    devices = find_open_devices()

    dynapcnn_network = get_demo_dynapcnn_network()
    for device_name, _ in devices.items():
        dynapcnn_network.to(device=device_name)

        visualizer = DynapcnnVisualizer(
            dvs_shape=dvs_shape, spike_collection_interval=spike_collection_interval
        )
        visualizer.create_visualizer_process(visualizer_id=visualizer_id)

        streamer_graph = samna.graph.EventFilterGraph()
        # Streamer graph
        # Dvs node
        (_, dvs_member_filter, _, streamer_node) = streamer_graph.sequential(
            [
                # samna.graph.JitSource(samna.speck2e.event.OutputEvent),
                dynapcnn_network.samna_device.get_model_source_node(),
                samna.graph.JitMemberSelect(),
                samna.graph.JitDvsEventToViz(samna.ui.Event),
                "VizEventStreamer",
            ]
        )
