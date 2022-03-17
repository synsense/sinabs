from dataclasses import dataclass
import math
import torch


@dataclass
class Heaviside:
    """
    Heaviside surrogat gradient with optional shift.

    Parameters:
        window:
            Distance between step of Heaviside surrogate gradient and
            threshold, relative to threshold.
    """

    window: float = 1.0

    def __call__(self, v_mem, threshold):
        return ((v_mem >= (threshold - self.window)).float()) / threshold


@dataclass
class MultiGaussian:
    """
    Surrogate gradient as defined in Yin et al., 2021.
    https://www.biorxiv.org/content/10.1101/2021.03.22.436372v2
    """

    mu: float = 0.0
    sigma: float = 0.5
    grad_scale: float = 1.0

    def __call__(self, v_mem, threshold):
        return (
            torch.exp(-(((v_mem - threshold) - self.mu) ** 2) / (2 * self.sigma**2))
            / torch.sqrt(2 * torch.tensor(math.pi))
            / self.sigma
        ) * self.grad_scale


@dataclass
class SingleExponential:
    """
    Surrogate gradient as defined in Shrestha and Orchard, 2018
    https://papers.nips.cc/paper/2018/hash/82f2b308c3b01637c607ce05f52a2fed-Abstract.html
    """

    grad_width: float = 0.5
    grad_scale: float = 1.0

    def __call__(self, v_mem, threshold):
        abs_width = threshold * self.grad_width
        return (
            self.grad_scale
            / abs_width
            * torch.exp(-torch.abs(v_mem - threshold) / abs_width)
        )
