import torch
import torch.nn as nn
from typing import Union, Tuple, List, Callable, Any, Dict, Optional
from torchview import ComputationGraph
from torchview import RecorderTensor
from torchview.recorder_tensor import Recorder
import warnings


def named_modules_map(
    model: nn.Module, model_name: str = "model"
) -> Dict[str, nn.Module]:
    """Inverse of named modules dictionary

    Args:
        model (nn.Module): The module to be hashed

    Returns:
        Dict[str, nn.Module]: A dictionary with modules as keys, and names as values
    """
    modules_map = {}
    for name, mod in model.named_modules():
        modules_map[mod] = name
    modules_map[model] = model_name
    return modules_map


class Node:
    def __init__(
        self,
        elem: Any,
        name: str,
        outgoing_nodes: Optional[List["Node"]] = None,
    ) -> None:
        self.elem = elem
        self.name = name
        if not outgoing_nodes:
            self.outgoing_nodes = []
        else:
            self.outgoing_nodes = outgoing_nodes

    def add_outgoing(self, node: "Node"):
        self.outgoing_nodes.append(node)

    def __str__(self) -> str:
        return f"Node: {self.name}, Out: {len(self.outgoing_nodes)}"

    def __eq__(self, other: Any) -> bool:
        # Two nodes are meant to be the same if they refer to the same element
        try:
            return self.elem is other.elem
        except AttributeError:
            return False

    def __hash__(self):
        # Two nodes are same if they reference the same element
        return hash(self.elem)


class Graph:
    def __init__(self, mod: nn.Module) -> None:
        self.elem_list = []
        self.node_map: Dict[Node, str] = {}
        self.modules_map = named_modules_map(mod)
        self.tensor_id_list = []

    @property
    def node_map_by_id(self):
        return {v: k for k, v in self.node_map.items()}

    def get_unique_tensor_id(self):
        if not self.tensor_id_list:
            self.tensor_id_list.append(0)
            return 0
        else:
            self.tensor_id_list.append(self.tensor_id_list[-1] + 1)
            return str(self.tensor_id_list[-1] + 1)

    def __contains__(self, elem: Union[torch.Tensor, nn.Module]):
        for elem_in_list in self.elem_list:
            if elem is elem_in_list:
                return True
        return False

    def add_elem(self, elem, name: str):
        if elem in self:
            warnings.warn(f"{name}: Node already exists for this element ")
            return self.find_node(elem)
        else:
            node = Node(elem, name)
            self.elem_list.append(elem)
            self.node_map[node] = name
            return node

    def add_or_get_node_for_elem(self, elem: Union[torch.Tensor, nn.Module]):
        if elem in self:
            return self.find_node(elem)
        else:
            # Generate a name
            if elem in self.modules_map:
                name = self.modules_map[elem]
            else:
                assert isinstance(elem, torch.Tensor)
                name = f"Tensor_{self.get_unique_tensor_id()}"
            # add and return the node
            new_node = self.add_elem(elem, name)
            return new_node

    def find_node(self, elem: Union[torch.Tensor, nn.Module]):
        for node in self.node_map.keys():
            if elem is node.elem:
                return node
        raise ValueError("elem not found")

    def add_edge(
        self,
        source: Union[torch.Tensor, nn.Module],
        destination: Union[torch.Tensor, nn.Module],
    ):
        source_node = self.add_or_get_node_for_elem(source)
        destination_node = self.add_or_get_node_for_elem(destination)
        source_node.add_outgoing(destination_node)
        return source_node, destination_node

    def __str__(self) -> str:
        return "\n".join([f"{n}" for n in self.node_map.keys()])

    def to_md(self) -> str:
        mermaid_md = """
```mermaid
graph TD;
"""
        for node, _ in self.node_map.items():
            for outgoing in node.outgoing_nodes:
                mermaid_md += f"{node.name} --> {outgoing.name};\n"
        end = """
```
"""
        return mermaid_md + end


_torch_module_call = torch.nn.Module.__call__


def module_forward_wrapper(model_graph: Graph) -> Callable[..., Any]:
    def my_forward(mod: nn.Module, *args, **kwargs) -> Any:
        # Iterate over all inputs
        for i, input_data in enumerate(args):
            # Create nodes and edges
            model_graph.add_edge(input_data, mod)
        out = _torch_module_call(mod, *args, **kwargs)
        if isinstance(out, tuple):
            out_tuple
        elif isinstance(out, torch.Tensor):
            out_tuple = (out,)
        else:
            raise Exception("Unknown output format")
        # Iterate over all outputs and create nodes and edges
        for i, output_data in enumerate(out_tuple):
            # Create nodes and edges
            model_graph.add_edge(mod, output_data)
        return out

    return my_forward


class GraphTracer:
    """
    Context manager to trace a model's execution graph

    Example:

    ```python
    with GraphTracer(mymodel) as tracer, torch.no_grad():
        out = mymodel(data)

    print(tracer.graph.to_md())
    ```
    """

    def __init__(self, mod: nn.Module) -> None:
        self.original_torch_call = nn.Module.__call__
        self.graph = Graph(mod)

    def __enter__(self) -> "GraphTracer":
        # Override the torch call method
        nn.Module.__call__ = module_forward_wrapper(self.graph)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        # Restore normal behavior
        nn.Module.__call__ = self.original_torch_call
