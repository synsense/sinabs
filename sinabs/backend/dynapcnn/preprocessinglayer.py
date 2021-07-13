import torch.nn as nn
from typing import Tuple, Optional
from sinabs.layers import Cropping2dLayer
from sinabs.layers import SumPool2d


class PreProcessingLayer(nn.Module):
    """"""

    def __init__(self, pool: (int, int) = (1, 1), crop: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None):
        super().__init__()
        if crop is not None:
            self._crop_layer = Cropping2dLayer(crop)
        else:
            self._crop_layer = None
        self._pool_layer = SumPool2d(pool)

        self._config_dict = {}
        self._update_config_dict()

    def _update_config_dict(self):
        raise NotImplementedError


    @property
    def pool_layer(self):
        return self._pool_layer

    @property
    def crop_layer(self):
        return self._crop_layer

    @property
    def output_shape(self):
        raise NotImplementedError


