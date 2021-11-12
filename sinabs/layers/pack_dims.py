from abc import ABC
from typing import Type, Optional

from torch import nn


class Squeeze(ABC):
    """
    Abstract base class to indicate that a layer is of "squeeze" type, i.e.
    its forward method expects data with batch and time as one dimension.
    """

    pass


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

    class Sqz(cls):
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
        ...as well as all parameters from {cls.__name__} layer

        """

        # New dict to prevent sharing object with parent classs
        if hasattr(cls, "external_backends"):
            external_backends = {
                name: squeeze_class(backend)
                for name, backend in cls.external_backends.items()
            }
        else:
            external_backends = dict()

        def __init__(
            self,
            num_timesteps: Optional[int] = None,
            batch_size: Optional[int] = None,
            *args,
            **kwargs,
        ):
            if num_timesteps is not None:
                self._num_timesteps = num_timesteps
                self._batch_size = None
            elif batch_size is not None:
                self._batch_size = batch_size
                self._num_timesteps = None
            else:
                raise TypeError(
                    "Either `num_timesteps` or `batch_size` must be provided as integer."
                )

            # Pass `num_timesteps` for slayer based layers to function correctly
            super().__init__(*args, **kwargs, num_timesteps=num_timesteps)

        def forward(self, data, *args, **kwargs):
            original_shape = data.shape

            # Unsqueeze batch and time
            if self._num_timesteps is not None:
                if data.shape[0] % self._num_timesteps != 0:
                    raise ValueError(
                        f"First dimension of `data` must be multiple of {self._num_timesteps}."
                    )
                unsqueezed = data.reshape(-1, self._num_timesteps, *original_shape[1:])
            else:
                if data.shape[0] % self._batch_size != 0:
                    raise ValueError(
                        f"First dimension of `data` must be multiple of {self._batch_size}."
                    )
                unsqueezed = data.reshape(self._batch_size, -1, *original_shape[1:])

            # Apply actual forward call
            output_unsqueezed = super().forward(unsqueezed)

            # Squeeze batch and time again
            return output_unsqueezed.reshape(-1, *output_unsqueezed.shape[2:])

        forward.__doc__ = f"""
        Same as :py:class:`{cls.__name__}`.forward but expects and returns batch
        and time as a single dimension: (batch x time, ...)"""

        @property
        def _param_dict(self) -> dict:
            """
            Dict of all parameters relevant for creating a new instance with same
            parameters as `self`
            """
            param_dict = super()._param_dict
            param_dict.update(
                batch_size=self._batch_size, num_timesteps=self._num_timesteps
            )
            return param_dict

        @property
        def num_timesteps(self):
            return self._num_timesteps

        @property
        def batch_size(self):
            return self._batch_size

        __qualname__ = cls.__name__ + "Squeeze"

    Sqz.__name__ = cls.__name__ + "Squeeze"

    Squeeze.register(Sqz)

    return Sqz
