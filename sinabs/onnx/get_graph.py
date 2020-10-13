import onnx
import torch
from sinabs import Network


def print_graph(graph: onnx.GraphProto):
    params = [x.name for x in graph.initializer]
    for i, node in enumerate(graph.node):
        inputs = list(filter(lambda x: x not in params, node.input))
        print(f"{node.name}: Inputs({inputs}), Outputs({node.output}")


def print_onnx_model(onnx_model: onnx.ModelProto):
    print_graph(onnx_model.graph)


def get_graph(model: Network, inputs):
    """
    Extract graph from a sinabs Network model

    :param model: sinabs.Netowrk model to extract the graph for
    :param inputs: Input tensor to extract graph
    :return: onnx.Graph
    """

    fname = "ann_temp.onnx"
    if len(inputs.shape) == 5:
        inputs = inputs.sum(0)
    torch.onnx.export(model.analog_model, inputs, fname)
    onnx_model = onnx.load(fname)
    return onnx_model.graph
