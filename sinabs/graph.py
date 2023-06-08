import torch
import torchview
import torch.nn as nn
from typing import Union, Tuple, List, Callable, Any, Dict, Optional
from torchview import ComputationGraph
import torchview.torchview as tw
from torchview import RecorderTensor
from torchview.recorder_tensor import Recorder
from torchview import TensorNode
from torchview.computation_node import NodeContainer
import graphviz
import warnings


class Node:
    def __init__(
        self,
        elem: Any,
        name: str,
        incoming_nodes: Optional[List["Node"]] = None,
        outgoing_nodes: Optional[List["Node"]] = None,
    ) -> None:
        self.elem = elem
        self.name = name
        # Initialize if None
        if not incoming_nodes:
            self.incoming_nodes = []
        else:
            self.incoming_nodes = incoming_nodes
        # Initialize if None
        if not outgoing_nodes:
            self.outgoing_nodes = []
        else:
            self.outgoing_nodes = outgoing_nodes

    def add_incoming(self, node: "Node"):
        self.incoming_nodes.append(node)

    def add_outgoing(self, node: "Node"):
        self.outgoing_nodes.append(node)

    def __str__(self) -> str:
        return f"Node: {self.name}, I: {len(self.incoming_nodes)}, O: {len(self.outgoing_nodes)}"

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


    def add_edge(self, source: Union[torch.Tensor, nn.Module], destination: Union[torch.Tensor, nn.Module]):
        source_node = self.add_or_get_node_for_elem(source)
        destination_node = self.add_or_get_node_for_elem(destination)
        print(f"Adding edge {source_node.name} -> {destination_node.name}")
        source_node.add_outgoing(destination_node)
        destination_node.add_incoming(source_node)
        return source_node, destination_node
    
    def __str__(self) -> str:
        return "\n".join([f"{n}" for n in self.node_map.keys()])

    def to_md(self)-> str:
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



def process_input(input_data: torch.Tensor):
    # Note: Works only when the input is a single tensor
    # Convert to recorder tensor
    recorder_tensor = input_data.as_subclass(RecorderTensor)
    # Create a corresponding node for it
    input_node = TensorNode(tensor=recorder_tensor, depth=0, name="Input")
    recorder_tensor.tensor_nodes = [input_node]
    return recorder_tensor, {}, NodeContainer(recorder_tensor.tensor_nodes)


_torch_module_call = torch.nn.Module.__call__


def module_forward_wrapper(model_graph: Graph) -> Callable[..., Any]:
    def _my_forward(mod: nn.Module, *args, **kwargs) -> Any:
        # Iterate over all inputs
        for i, input_data in enumerate(args):
            # Create nodes and edges
            model_graph.add_edge(input_data, mod)
        out = _torch_module_call(mod, *args, **kwargs)
        if isinstance(out, tuple):
            out_tuple
        elif isinstance(out, torch.Tensor):
            out_tuple = out,
        else:
            raise Exception("Unknown output format")
        # Iterate over all outputs and create nodes and edges
        for i, output_data in enumerate(out_tuple):
            # Create nodes and edges
            model_graph.add_edge(mod, output_data)
        return out

    return _my_forward


def forward_prop(
    model: nn.Module, input_data: RecorderTensor, model_graph: ComputationGraph
):
    model.eval()
    model = model.to("cpu")
    new_module_forward = module_forward_wrapper(model_graph)
    with Recorder(_torch_module_call, new_module_forward, model_graph):
        model(input_data)
    return


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


def extract_graph(model: nn.Module, input_data: torch.Tensor) -> Graph:
    # Modify the input somehow
    input_record_tensor, kwargs_record_tensor, input_nodes = process_input(input_data)
    # Create a graph
    visual_graph = graphviz.Digraph(
        name="..", engine="dot", strict=True, filename="somefile.dot"
    )
    model_graph = ComputationGraph(
        visual_graph=visual_graph, root_container=input_nodes
    )

    # Populate it
    forward_prop(model, input_record_tensor, model_graph=model_graph)

    model_graph.fill_visual_graph()

    return model_graph
