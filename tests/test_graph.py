import torch
import torch.nn as nn


class Add(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, data1, data2):
        return data1 + data2


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
