from typing import Union, Optional
import torch
from .lif import LIF
from .reshape import SqueezeMixin


class ExpLeak(LIF):
    """
    A Leaky Integrator layer.

    Neuron dynamics in discrete time:

    .. math ::
        V_{mem}(t+1) = \\alpha V_{mem}(t) + (1-\\alpha)\\sum z(t)

    where :math:`\\alpha =  e^{-1/tau_{mem}}` and :math:`\\sum z(t)` represents the sum of all input currents at time :math:`t`.

    Parameters
    ----------
    tau_leak: float
        Membrane potential time constant.
    threshold_low: float or None
        Lower bound for membrane potential v_mem, clipped at every time step.
    train_alphas: bool
        When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself.
    shape: torch.Size
        Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
    norm_input: bool
        When True, normalise input current by tau. This helps when training time constants.
    """

    def __init__(
        self,
        tau_leak: Union[float, torch.Tensor],
        shape: Optional[torch.Size] = None,
        train_alphas: bool = False,
        threshold_low: Optional[float] = None,
        norm_input: bool = False,
    ):
        super().__init__(
            tau_mem=tau_leak,
            tau_syn=None,
            train_alphas=train_alphas,
            threshold_low=threshold_low,
            shape=shape,
            activation_fn=None,
            norm_input=norm_input,
        )

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(tau_leak=self.tau_mem)
        param_dict.pop('tau_mem')
        return param_dict

class ExpLeakSqueeze(ExpLeak, SqueezeMixin):
    """
    Same as parent ExpLeak class, only takes in squeezed 4D input (Batch*Time, Channel, Height, Width)
    instead of 5D input (Batch, Time, Channel, Height, Width) in order to be compatible with
    layers that can only take a 4D input, such as convolutional and pooling layers.
    """

    def __init__(self, batch_size=None, num_timesteps=None, **kwargs):
        super().__init__(**kwargs)
        self.squeeze_init(batch_size, num_timesteps)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.squeeze_forward(input_data, super().forward)

    @property
    def _param_dict(self) -> dict:
        return self.squeeze_param_dict(super()._param_dict)
