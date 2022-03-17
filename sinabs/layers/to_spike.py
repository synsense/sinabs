from torch import nn
from typing import Tuple
import torch


class Img2SpikeLayer(nn.Module):
    """
    Layer to convert images to Spikes
    """

    def __init__(
        self,
        image_shape,
        tw: int = 100,
        max_rate: float = 1000,
        norm: float = 255.0,
        squeeze: bool = False,
        negative_spikes: bool = False,
    ):
        """
        Layer converts images to spikes

        :param image_shape: tuple image shape
        :param tw: int Time window length
        :param max_rate: maximum firing rate of neurons
        :param layer_name: string layer name
        :param norm: the supposed maximum value of the input (default 255.0)
        :param squeeze: whether to remove singleton dimensions from the input
        :param negative_spikes: whether to allow negative spikes in response \
        to negative input
        """
        super().__init__()
        self.tw = tw
        self.max_rate = max_rate
        self.norm = norm
        self.squeeze = squeeze
        self.negative_spikes = negative_spikes

    def forward(self, img_input):
        if self.squeeze:
            img_input = img_input.squeeze()
        random_tensor = torch.rand(self.tw, *tuple(img_input.shape)).to(
            img_input.device
        )
        if not self.negative_spikes:
            firing_probs = (img_input / self.norm) * (self.max_rate / 1000)
            spk_img = (random_tensor < firing_probs).float()
        else:
            firing_probs = (img_input.abs() / self.norm) * (self.max_rate / 1000)
            spk_img = (random_tensor < firing_probs).float() * img_input.sign().float()
        self.spikes_number = spk_img.abs().sum()
        self.tw = len(spk_img)
        return spk_img

    def get_output_shape(self, input_shape: Tuple):
        # The time dimension is not included in the shape
        # NOTE: This is not true if the squeeze is false but input_shape has a batch_size
        # TODO: Fix this
        return input_shape  # (self.tw, *input_shape)


class Sig2SpikeLayer(torch.nn.Module):
    """
    Layer to convert analog Signals to Spikes
    """

    def __init__(
        self,
        channels_in,
        tw: int = 1,
        norm_level: float = 1,
        spk_out: bool = True,
    ):
        """
        Layer converts analog signals to spikes

        :param channels_in: number of channels in the analog signal
        :param tw: int number of time steps for each sample of the signal (up sampling)
        :param layer_name: string layer name

        """
        super().__init__()
        self.tw = tw
        self.norm_level = norm_level
        self.spk_out = spk_out

    def get_output_shape(self, input_shape: Tuple):
        channels, time_steps = input_shape
        return (self.tw * time_steps, channels)

    def forward(self, signal):
        """
        Convert a signal to the corresponding spikes

        :param signal: [Channel, Sample(t)]
        :return:
        """
        channels, time_steps = signal.shape
        if self.tw != 1:
            signal = signal.view(-1, 1).repeat(1, self.tw).view(channels, -1)
        signal = signal.transpose(1, 0)
        if self.spk_out:
            random_tensor = (
                torch.rand(self.tw * time_steps, channels).to(signal.device)
                * self.norm_level
            )
            spk_sig = (random_tensor < signal).float()
        else:
            # If there is no conversion to spikes
            # just replicate the signal as current injection
            spk_sig = signal

        self.spikes_number = spk_sig.abs().sum()
        return spk_sig
