import torch
from typing import Callable, Optional


class SqueezeMixin:
    def squeeze_init(self, batch_size: Optional[int], num_timesteps: Optional[int]):
        if not batch_size and not num_timesteps:
            raise TypeError("You need to specify either batch_size or num_timesteps.")
        if not batch_size:
            batch_size = -1 
        if not num_timesteps:
            num_timesteps = -1
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps

    def squeeze_forward(self, input_data: torch.Tensor, forward_method: Callable):
        inflated_input = input_data.reshape(self.batch_size, self.num_timesteps, *input_data.shape[1:])
        inflated_output = forward_method(inflated_input)
        return inflated_output.flatten(start_dim=0, end_dim=1)

    def squeeze_param_dict(self, param_dict: dict) -> dict:
        param_dict.update(
            batch_size=self.batch_size,
            num_timesteps=self.num_timesteps,
        )
        return param_dict