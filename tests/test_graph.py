import torch
import torch.nn as nn


# Branched model
class MyBranchedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu1 = nn.ReLU()
        self.relu2_1 = nn.ReLU()
        self.relu2_2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, data):
        out1 = self.relu1(data)
        out2_1 = self.relu2_1(out1)
        out2_2 = self.relu2_2(out1)
        out3 = self.relu3(out2_1 + out2_2)
        out3.foo = "foo"
        return out3


input_shape = (2, 28, 28)
batch_size = 1

data = torch.ones((batch_size, *input_shape))

mymodel = MyBranchedModel()


def test_named_modules_map():
    from sinabs.graph import named_modules_map

    mod_map = named_modules_map(mymodel)
    print(mod_map)


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

    print(model_graph.to_md())


def test_graph_tracer():
    from sinabs.graph import GraphTracer, named_modules_map

    with GraphTracer(named_modules_map(mymodel)) as tracer, torch.no_grad():
        out = mymodel(data)

    print(tracer.graph.to_md())
