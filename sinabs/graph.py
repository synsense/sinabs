import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn


def named_modules_map(
    model: nn.Module, model_name: Optional[str] = "model"
) -> Dict[str, nn.Module]:
    """Inverse of named modules dictionary

    Args:
        model (nn.Module): The module to be hashed
        model_name (str | None): Name of the top level module. If this doesn't need to be include, this option can be set to None

    Returns:
        Dict[str, nn.Module]: A dictionary with modules as keys, and names as values
    """
    modules_map = {}
    for name, mod in model.named_modules():
        modules_map[mod] = name
    if model_name is None:
        del modules_map[model]
    else:
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
    def __init__(self, module_names: Dict[nn.Module, str]) -> None:
        self.module_names = module_names
        self.elem_list = []
        self.node_list: List[Node] = []
        self._last_used_tensor_id = None

    @property
    def node_map_by_id(self):
        return {n.name: n for n in self.node_list}

    def num_edges(self) -> int:
        count = 0
        for node in self.node_list:
            count += node.outgoing_nodes
        return count

    def get_unique_tensor_id(self) -> str:
        if self._last_used_tensor_id is None:
            self._last_used_tensor_id = 0
        else:
            self._last_used_tensor_id += 1
        return str(self._last_used_tensor_id)

    def __contains__(self, elem: Union[torch.Tensor, nn.Module]) -> bool:
        for elem_in_list in self.elem_list:
            if elem is elem_in_list:
                return True
        return False

    def add_elem(self, elem, name: str) -> Node:
        if elem in self:
            warnings.warn(f"{name}: Node already exists for this element ")
            return self.find_node(elem)
        else:
            node = Node(elem, name)
            self.elem_list.append(elem)
            self.node_list.append(node)
            return node

    def add_or_get_node_for_elem(self, elem: Union[torch.Tensor, nn.Module]):
        if elem in self:
            return self.find_node(elem)
        else:
            # Generate a name
            if elem in self.module_names:
                name = self.module_names[elem]
            else:
                assert isinstance(elem, torch.Tensor)
                name = f"Tensor_{self.get_unique_tensor_id()}{tuple(elem.shape)}"
            # add and return the node
            new_node = self.add_elem(elem, name)
            return new_node

    def find_node(self, elem: Union[torch.Tensor, nn.Module]):
        for node in self.node_list:
            if elem is node.elem:
                return node
        raise ValueError("elem not found")

    def add_edge(
        self,
        source: Union[torch.Tensor, nn.Module],
        destination: Union[torch.Tensor, nn.Module],
    ):
        if self._is_mod_and_not_in_module_names(source): return
        if self._is_mod_and_not_in_module_names(destination): return

        source_node = self.add_or_get_node_for_elem(source)
        destination_node = self.add_or_get_node_for_elem(destination)
        source_node.add_outgoing(destination_node)
        return source_node, destination_node

    def get_leaf_modules(self) -> Dict[nn.Module, str]:
        filtered_module_names = {}

        for mod, _ in self.module_names.items():
            # Add module to dict
            filtered_module_names[mod] = self.module_names[mod]
            child_in_graph = False
            for _, submod in mod.named_children():
                if submod in self:
                    child_in_graph = True
                    break
            if child_in_graph:
                del filtered_module_names[mod]
        return filtered_module_names

    def _is_mod_and_not_in_module_names(self, elem: Any) -> bool:
        """Check if a node is a module and is included in the module_names of this graph

        Args:
            node (Node): Node to verify

        Returns:
            bool
        """
        if isinstance(elem, nn.Module) and elem not in self.module_names:
            return True
        else:
            return False

    def populate_from(self, other_graph: "Graph"):
        for node in other_graph.node_list:
            for outgoing_node in node.outgoing_nodes:
                self.add_edge(node.elem, outgoing_node.elem)

    def __str__(self) -> str:
        return self.to_md()

    def to_md(self) -> str:
        mermaid_md = """
```mermaid
graph TD;
"""
        for node in self.node_list:
            for outgoing in node.outgoing_nodes:
                mermaid_md += f"{node.name} --> {outgoing.name};\n"
        end = """
```
"""
        return mermaid_md + end

    def leaf_only(self) -> "Graph":
        leaf_modules = self.get_leaf_modules()
        filtered_graph = Graph(leaf_modules)
        # Populate edges
        filtered_graph.populate_from(self)
        return filtered_graph

    def ignore_submodules_of(self, classes: List[Type]) -> "Graph":
        new_named_modules = {}

        # Gather a list of all top level modules, whose submodules are to be ignored
        top_level_modules: List[nn.Module] = []
        for mod in self.module_names.keys():
            if mod.__class__ in classes:
                top_level_modules.append(mod)

        # List all the submodules of the above module list
        sub_modules_to_ignore: List[nn.Module] = []
        for top_mod in top_level_modules:
            for sub_mod in top_mod.modules():
                if sub_mod is not top_mod:
                    sub_modules_to_ignore.append(sub_mod)

        # Iterate over all modules and check if they are submodules of the above list
        for mod, name in self.module_names.items():
            if mod not in sub_modules_to_ignore:
                new_named_modules[mod] = name
        # Create a new graph with the allowed modules
        new_graph = Graph(new_named_modules)
        new_graph.populate_from(self)
        return new_graph


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



def extract_graph(model: nn.Module, sample_data: Any, model_name: Optional[str] = "model")->Graph:
    with GraphTracer(named_modules_map(model, model_name=model_name)) as tracer, torch.no_grad():
        out = model(sample_data)

    return tracer.graph