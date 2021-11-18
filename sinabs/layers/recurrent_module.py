import torch
from typing import Type


def recurrent_class(cls: Type) -> Type:

    if not hasattr(cls, "forward"):
        return ValueError("`cls` must be a layer class with forward method.")

    class RecurrentModule(cls):
        def __init__(self, 
                     rec_connectivity: torch.nn.Module,
                     *args,
                     **kwargs,
            ):
            super().__init__(*args, **kwargs)
            self.rec_connectivity = rec_connectivity


        def forward(self, input_current: torch.Tensor):
            """
            Helper loop to add recurrent input to forward input.

            Parameters
            ----------
            input_current : torch.Tensor
                Data to be processed. Expected shape: (batch, time, ...)

            Returns
            -------
            torch.Tensor
                Output data. Same shape as `input_spikes`.
            """

            batch_size, n_time_steps, *other_dimensions = input_current.shape
            rec_out = torch.zeros((batch_size, 1, *other_dimensions))
            output_spikes = torch.zeros_like(input_current)

            for step in range(n_time_steps):
                total_input = input_current[:, step:step+1] + rec_out

                # compute output spikes
                output = super().forward(total_input)
                output_spikes[:, step:step+1] = output

                # compute recurrent output that will be added to the input at the next time step
                rec_out = self.rec_connectivity(output).reshape(input_current[:, step:step+1].shape)

            return output_spikes

    RecurrentModule.__name__ = cls.__name__ + "Recurrent"

    return RecurrentModule
    