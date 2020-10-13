from pathlib import Path

MODELS_FOLDER = Path(__file__).resolve().parent / "models"


def test_threshold_subtract_onnx_eq():
    import torch
    import numpy as np

    from sinabs.layers.functional import threshold_subtract

    class ThresholdTest(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inp):
            return threshold_subtract(inp, 1.0)

    inp = (torch.rand(10)-0.5)*3
    model = ThresholdTest()

    torch_out = model(inp)
    print(inp, "\nTorch:", torch_out)
    fname = MODELS_FOLDER / "threshold_subtract.onnx"

    torch.onnx.export(model, (inp,), fname, export_params=True,
                      input_names=["t_input"],
                      output_names=["t_output"],
                      dynamic_axes={"t_input": [0], "t_output": [0]},
                      verbose=False,
                      )

    import onnxruntime

    session = onnxruntime.InferenceSession(str(fname))

    ort_inputs = {session.get_inputs()[0].name: inp.numpy()}
    ort_outs = session.run(None, ort_inputs)
    print("Onnx:", ort_outs[0])

    np.testing.assert_allclose(torch_out.detach().numpy(), ort_outs[0])
