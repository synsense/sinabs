import torch
import torch.nn as nn

from sinabs.layers import Add


# Branched model
class MyBranchedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.relu2_1 = nn.ReLU()
        self.relu2_2 = nn.ReLU()
        self.add_mod = Add()
        self.relu3 = nn.ReLU()

    def forward(self, data):
        out1 = self.relu1(data)
        out2_1 = self.relu2_1(out1)
        out2_2 = self.relu2_2(out1)
        out3 = self.add_mod(out2_1, out2_2)
        out4 = self.relu3(out3)
        return out4


input_shape = (2, 28, 28)
batch_size = 1

data = torch.ones((batch_size, *input_shape))

mymodel = MyBranchedModel()


class DeepModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block1 = MyBranchedModel()
        self.block2 = MyBranchedModel()

    def forward(self, data):
        out = self.block1(data)
        out2 = self.block2(out)
        return out2


mydeepmodel = DeepModel()


def test_named_modules_map():
    from sinabs.graph import named_modules_map

    mod_map = named_modules_map(mymodel)
    print(mod_map)
    for k, v in mod_map.items():
        assert isinstance(k, nn.Module)
        assert isinstance(v, str)


def test_module_forward_wrapper():
    mymodel = MyBranchedModel()

    orig_call = nn.Module.__call__

    from sinabs.graph import Graph, module_forward_wrapper, named_modules_map

    model_graph = Graph(named_modules_map(mymodel))
    new_call = module_forward_wrapper(model_graph)

    # Override call to the new wrapped call
    nn.Module.__call__ = new_call

    with torch.no_grad():
        out = mymodel(data)

    # Restore normal behavior
    nn.Module.__call__ = orig_call

    print(model_graph)
    assert (
        len(model_graph.node_list) == 1 + 5 + 5 + 1
    )  # 1 top module + 5 submodules + 5 tensors + 1 output tensor


def test_graph_tracer():
    from sinabs.graph import GraphTracer, named_modules_map

    with GraphTracer(named_modules_map(mymodel)) as tracer, torch.no_grad():
        out = mymodel(data)

    print(tracer.graph)
    assert (
        len(tracer.graph.node_list) == 1 + 5 + 5 + 1
    )  # 1 top module + 5 submodules + 5 tensors + 1 output tensor


def test_leaf_only_graph():
    from sinabs.graph import GraphTracer, named_modules_map

    with GraphTracer(named_modules_map(mydeepmodel)) as tracer, torch.no_grad():
        out = mydeepmodel(data)

    print(tracer.graph)

    # Get graph with just the leaf nodes
    leaf_graph = tracer.graph.leaf_only()
    print(leaf_graph)
    assert (
        len(leaf_graph.node_list) == len(tracer.graph.node_list) - 3
    )  # No more top modules


def test_ignore_submodules_of():
    from sinabs.graph import GraphTracer, named_modules_map

    with GraphTracer(named_modules_map(mydeepmodel)) as tracer, torch.no_grad():
        out = mydeepmodel(data)

    top_overview_graph = tracer.graph.ignore_submodules_of(
        [MyBranchedModel]
    ).leaf_only()
    print(top_overview_graph)
    assert len(top_overview_graph.node_list) == 2 + 2 + 1


def test_snn_branched():
    from sinabs.layers import IAFSqueeze, ConcatenateChannel, SumPool2d
    from torch.nn import Conv2d
    from sinabs.graph import extract_graph

    class MySNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = Conv2d(2, 8, 3, bias=False)
            self.iaf1 = IAFSqueeze(batch_size=1)
            self.pool1 = SumPool2d(2)
            self.conv2_1 = Conv2d(8, 16, 3, stride=1, padding=1, bias=False)
            self.iaf2_1 = IAFSqueeze(batch_size=1)
            self.pool2_1 = SumPool2d(2)
            self.conv2_2 = Conv2d(8, 16, 5, stride=1, padding=2, bias=False)
            self.iaf2_2 = IAFSqueeze(batch_size=1)
            self.pool2_2 = SumPool2d(2)
            self.concat = ConcatenateChannel()
            self.conv3 = Conv2d(32, 10, 3, stride=3, bias=False)
            self.iaf3 = IAFSqueeze(batch_size=1)

        def forward(self, spikes):
            out = self.conv1(spikes)
            out = self.iaf1(out)
            out = self.pool1(out)

            out1 = self.conv2_1(out)
            out1 = self.iaf2_1(out1)
            out1 = self.pool2_1(out1)

            out2 = self.conv2_2(out)
            out2 = self.iaf2_2(out2)
            out2 = self.pool2_2(out2)

            out = self.concat(out1, out2)
            out = self.conv3(out)
            out = self.iaf3(out)
            return out

    my_snn = MySNN()
    graph = extract_graph(
        my_snn, sample_data=torch.rand((100, 2, 14, 14)), model_name=None
    )

    print(graph)
