import os
import pytest
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer


def X_available() -> bool:
    from subprocess import PIPE, Popen

    p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
    p.communicate()
    return p.returncode == 0


@pytest.mark.skipif(not X_available(), reason="A window needs to pop. Needs UI")
def test_visualizer_initialization():
    dvs_shape = (128, 128)
    spike_collection_interval = 500
    visualizer_id = 3

    visualizer = DynapcnnVisualizer(
        dvs_shape=dvs_shape, spike_collection_interval=spike_collection_interval
    )
    visualizer.create_visualizer_process(visualizer_id=visualizer_id)
