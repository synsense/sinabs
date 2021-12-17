# import torch
# import torch.nn as nn
# from pathlib import Path
# from sinabs.activation import ActivationFunction, MembraneSubtract

# MODELS_FOLDER = Path(__file__).resolve().parent / "models"


# def build_model():
#     from sinabs.from_torch import from_model

#     layers = [
#         nn.Conv2d(2, 8, kernel_size=2),
#         nn.AvgPool2d(2, stride=2),
#         nn.ReLU(),
#         nn.Conv2d(8, 10, kernel_size=3),
#         nn.AvgPool2d(3),
#         nn.ReLU(),
#     ]

#     ann = nn.Sequential(*layers)
#     net = from_model(ann)
#     return net


# def test_sinabs_model_to_onnx():
#     import onnx
#     from sinabs.onnx import print_onnx_model, get_graph, print_graph

#     net = build_model()
#     dummy_input = torch.zeros([1, 2, 64, 64])  # One time step
#     net(dummy_input)  # first pass to create all state variables
#     fname = MODELS_FOLDER / "snn.onnx"

#     torch.onnx.export(
#         net.spiking_model,
#         dummy_input,
#         fname,
#         input_names=["tchw_input"],
#         output_names=["tchw_output"],
#         dynamic_axes={"tchw_input": [0], "tchw_output": [0]},
#         verbose=False,
#     )
#     snn_model = onnx.load(fname)
#     ann_graph = get_graph(net, dummy_input)
#     print("................")
#     print_graph(ann_graph)
#     print("................")
#     print_onnx_model(snn_model)


# def test_graph_generation_ann():
#     import torchvision.models
#     import onnx
#     from sinabs.onnx import print_onnx_model

#     model = torchvision.models.resnet18()
#     dummy_input = torch.zeros([1, 3, 224, 224])
#     fname = MODELS_FOLDER / "resnet18.onnx"

#     torch.onnx.export(model, dummy_input, fname)
#     onnx_model = onnx.load(fname)

#     print_onnx_model(onnx_model)


# def test_onnx_sinabs_SpikingLayer():
#     import numpy as np
#     from sinabs.from_torch import from_model

#     layers = [
#         nn.ReLU(),
#     ]

#     ann = nn.Sequential(*layers)
#     net = from_model(ann)
#     inp = (torch.rand((50, 2, 32, 32)) > 0.5).float()
#     dummy = torch.zeros_like(inp)

#     net.spiking_model(dummy)

#     fname = MODELS_FOLDER / "test_spk.onnx"
#     torch.onnx.export(
#         net.spiking_model,
#         (dummy,),
#         fname,
#         export_params=True,
#         input_names=["tchw_input"],
#         output_names=["tchw_output"],
#         dynamic_axes={"tchw_input": [0], "tchw_output": [0]},
#         verbose=False,
#     )

#     # Run torch model
#     out_torch = net.spiking_model(inp)

#     # Run onnx model
#     import onnxruntime

#     session = onnxruntime.InferenceSession(str(fname))
#     ort_inputs = {session.get_inputs()[0].name: inp.numpy()}
#     ort_outs = session.run(None, ort_inputs)

#     np.testing.assert_allclose(
#         out_torch.detach().numpy(), ort_outs[0], rtol=1e-3, atol=1e-5
#     )


# def test_onnx_vs_sinabs_equivalence():
#     import numpy as np
#     net = build_model()

#     inp = (torch.rand((50, 2, 32, 32)) > 0.5).float()
#     dummy = torch.zeros_like(inp)

#     net.spiking_model(dummy)

#     fname = MODELS_FOLDER / "test_spk_net.onnx"
#     torch.onnx.export(
#         net.spiking_model,
#         (dummy,),
#         fname,
#         export_params=True,
#         input_names=["tchw_input"],
#         output_names=["tchw_output"],
#         dynamic_axes={"tchw_input": [0], "tchw_output": [0]},
#         verbose=False,
#     )

#     out_torch = net.spiking_model(inp)

#     import onnxruntime
#     session = onnxruntime.InferenceSession(str(fname))
#     ort_inputs = {session.get_inputs()[0].name: inp.numpy()}
#     ort_outs = session.run(None, ort_inputs)

#     np.testing.assert_allclose(
#         out_torch.detach().numpy(), ort_outs[0], atol=1e-5
#     )


# def test_threshold_subtract_onnx_eq():

#     class ThresholdTest(torch.nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.activation_fn = ActivationFunction(spike_threshold=1.,
#                                                     reset_fn=MembraneSubtract(),
#                                                    )

#         def forward(self, inp):
#             state = {'v_mem': inp}
#             return self.activation_fn(state)

#     inp = (torch.rand(10)-0.5)*3
#     model = ThresholdTest()

#     torch_out = model(inp)
#     print(inp, "\nTorch:", torch_out)
#     fname = MODELS_FOLDER / "threshold_subtract.onnx"

#     torch.onnx.export(model, (inp,), fname, export_params=True,
#                       input_names=["t_input"],
#                       output_names=["t_output"],
#                       dynamic_axes={"t_input": [0], "t_output": [0]},
#                       verbose=False,
#                       )

#     import onnxruntime

#     session = onnxruntime.InferenceSession(str(fname))

#     ort_inputs = {session.get_inputs()[0].name: inp.numpy()}
#     ort_outs = session.run(None, ort_inputs)
#     print("Onnx:", ort_outs[0])

#     np.testing.assert_allclose(torch_out.detach().numpy(), ort_outs[0])
