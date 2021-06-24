from abc import ABC
from typing import Type, Optional

from torch import nn


class Squeezed(ABC):
    """
    Abstract base class to indicate that a layer is of "squeezed" type, i.e.
    its forward method expects data with batch and time as one dimension.
    """

    pass


class SqueezeBatchTime(nn.Module):
    """
    Convenience layer that
    """

    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

    def forward(self, data):
        all_spikes = data.reshape((-1, *data.shape[2:]))
        return all_spikes


class UnsqueezeBatchTime(nn.Module):
    def __init__(self, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

    def forward(self, data):
        syn_out = data.reshape((self.batch_size, -1, *data.shape[1:]))
        return syn_out


def squeeze_class(cls: Type) -> Type:
    """
    Factory function to generate a subclass of a given class, whose forward
    method expects batch and time as a single dimension, splits them into
    separate dimensions and passes the data on to the parent's forward method.

    Parameters
    ----------
    cls : Type
        Class from which subclass is to be generated.

    Returns
    -------
    Type
        Subclass of `cls`, which accepts data in squeezed format.

    """

    if not hasattr(cls, "forward"):
        return ValueError("`cls` must be a layer class with forward method.")

    class Squeeze(cls):
        __doc__ = f"""
        "Squeezed" version of :py:class:`{cls.__name__}`, whose forward method expects
        batch and time as a single dimension.

        Parameters
        ----------
        num_timesteps : Optional[int]
            Number of timesteps per sample. Batch size will be inferred dynamically.
        batch_size: Optional[int]:
            Batch size of data. Ignored if `num_timesteps` is None, otherwise must
            be provided. Number of timesteps per sample will be inferred dynamically.
        """

        def __init__(
            self,
            num_timesteps: Optional[int] = None,
            batch_size: Optional[int] = None,
            *args,
            **kwargs,
        ):
            if num_timesteps is not None:
                self.num_timesteps = num_timesteps
                self.batch_size = None
            elif batch_size is not None:
                self.batch_size = batch_size
                self.num_timesteps = None
            else:
                raise TypeError(
                    "Either `num_timesteps` or `batch_size` must be provided as integer."
                )

            super().__init__(*args, **kwargs)

        def forward(self, data, *args, **kwargs):
            original_shape = data.shape

            # Unsqueeze batch and time
            if self.num_timesteps is not None:
                if data.shape[0] % self.num_timesteps != 0:
                    raise ValueError(
                        f"First dimension of `data` must be multiple of {self.num_timesteps}."
                    )
                unsqueezed = data.reshape(-1, self.num_timesteps, *original_shape[1:])
            else:
                if data.shape[0] % self.batch_size != 0:
                    raise ValueError(
                        f"First dimension of `data` must be multiple of {self.batch_size}."
                    )
                unsqueezed = data.reshape(self.batch_size, -1, *original_shape[1:])

            # Apply actual forward call
            output_unsqueezed = super().forward(unsqueezed)

            # Squeeze batch and time again
            return output_unsqueezed.reshape(-1, *output_unsqueezed.shape[2:])

        forward.__doc__ = f"""
        Same as :py:class:`{cls.__name__}`.forward but expects and returns batch
        and time as a single dimension: (batch x time, ...)"""

        def _param_dict(self) -> dict:
            """
            Dict of all parameters relevant for creating a new instance with same
            parameters as `self`
            """
            param_dict = super()._param_dict()
            param_dict.update(
                batch_size=self.batch_size, num_timesteps=self.num_timesteps
            )
            return param_dict

        __qualname__ = cls.__name__ + "Squeeze"

    Squeeze.__name__ = cls.__name__ + "Squeeze"

    return Squeeze
