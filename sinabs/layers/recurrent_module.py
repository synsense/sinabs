import torch


class RecurrentModule(torch.nn.Module):
    def __init__(self, 
                 layer: torch.nn.Module, 
                 rec_connectivity: torch.nn.Module,
        ):
        super().__init__()
        self.layer = layer
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
            output = self.layer(total_input)
            output_spikes[:, step:step+1] = output
            
            # compute recurrent output that will be added to the input at the next time step
            rec_out = self.rec_connectivity(output.reshape(batch_size, -1)).reshape(input_current[:, step:step+1].shape)

        return output_spikes

    def reset_states(self, shape=None, randomize=False):
        self.layer.reset_state(shape=shape, randomize=randomize)
        
    def zero_grad(self, set_to_none: bool = False):
        self.layer.zero_grad(set_to_none=set_to_none) 
    