from typing import Optional

import torch
import torch.nn as nn

from .stateful_layer import StatefulLayer


class Sequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__()
        self.stateful_layers = []
        for i, module in enumerate(args):
            self.add_module(str(i), module)
            if isinstance(module, StatefulLayer):
                self.stateful_layers.append(module)

    def forward(
        self, input_tensor: torch.tensor, state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        state = [None] * len(self) if state is None else state

        n_steps = input_tensor.shape[1]
        for step in range(n_steps):
            input_ = input_tensor[:, step]
            for i, module in enumerate(self):
                if module in self.stateful_layers:
                    input_, state[i] = module(input_, state[i])
                else:
                    input_ = module(input_)
