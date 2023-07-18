import samna
import numpy as np
import torch.nn as nn
import sinabs.layers as sl
from typing import List

def from_sinabs(my_network: nn.Sequential) -> "SpecksimNetwork":
    graph = samna.graph.EventFilterGraph()
    filters = []
    
    for layer in my_network:
        if isinstance(layer, nn.Conv2d):
            if layer.bias:
                raise ValueError("Biases are not supported!")
            ...
        elif isinstance(layer, (sl.SumPool2d, nn.AvgPool2d)):
            ...
        elif isinstance(layer, (sl.IAF, sl.IAFSqueeze)):
            ...
        else:
            raise TypeError("Only Conv2d, SumPool2d and IAF layers are supported.")
    
    graph.sequential([
        samna.BasicSourceNode_specksim_events_spike(), 
        *filters,
        samna.BasicSinkNode_specksim_events_spike() 
    ])
    
    return SpecksimNetwork(graph) 

class SpecksimNetwork:
    def __init__(
        self,
        graph: samna.graph.EventFilterGraph 
    ):
        self.network: samna.graph.EventFilterGraph = graph 
    
    
    def forward(self, xytp: np.record) -> np.record:
        # convert to specksim spikes
        self.network.start()
        # do the forward pass
        # convert output to record
        self.network.stop()
        # return
    
    def __call__(self, xytp: np.record) -> np.record:
        return self.forward(xytp)
    
    def xytp_to_specksim_spikes(self, xytp: np.record) -> List[samna.specksim.events.Spike]:
        ...
    
    def specksim_spikes_to_xytp(self, spikes: List[samna.specksim.events.Spike]) -> np.record:
        ...