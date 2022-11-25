import math
from dataclasses import dataclass

import torch


@dataclass
class Heaviside:
    """Heaviside surrogate gradient with optional shift.

    Parameters:
        window: Distance between step of Heaviside surrogate gradient and
                threshold, relative to threshold.
    """

    window: float = 1.0

    def __call__(self, v_mem, spike_threshold):
        return ((v_mem >= (spike_threshold - self.window)).float()) / spike_threshold


def gaussian(x: torch.Tensor, mu: float, sigma: float):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (
        sigma * torch.sqrt(2 * torch.tensor(math.pi))
    )


@dataclass
class Gaussian:
    """Gaussian surrogate gradient function.

    Parameters
        mu: The mean of the Gaussian.
        sigma: The standard deviation of the Gaussian.
        grad_scale: Scale the gradients arbitrarily.
    """

    mu: float = 0.0
    sigma: float = 0.5
    grad_scale: float = 1.0

    def __call__(self, v_mem, spike_threshold):
        return (
            gaussian(x=v_mem - spike_threshold, mu=self.mu, sigma=self.sigma)
            * self.grad_scale
        )


@dataclass
class MultiGaussian:
    """Surrogate gradient as defined in Yin et al., 2021.

    https://www.biorxiv.org/content/10.1101/2021.03.22.436372v2

    Parameters
        mu: The mean of the Gaussian.
        sigma: The standard deviation of the Gaussian.
        h: Controls the magnitude of the negative parts of the kernel.
        s: Controls the width of the negative parts of the kernel.
        grad_scale: Scale the gradients arbitrarily.
    """

    mu: float = 0.0
    sigma: float = 0.5
    h: float = 0.15
    s: float = 6
    grad_scale: float = 1.0

    def __call__(self, v_mem, spike_threshold):
        return (
            (1 + self.h)
            * gaussian(x=v_mem - spike_threshold, mu=self.mu, sigma=self.sigma)
            - self.h
            * gaussian(
                x=v_mem - spike_threshold, mu=self.sigma, sigma=self.s * self.sigma
            )
            - self.h
            * gaussian(
                x=v_mem - spike_threshold, mu=-self.sigma, sigma=self.s * self.sigma
            )
        ) * self.grad_scale


@dataclass
class SingleExponential:
    """Surrogate gradient as defined in Shrestha and Orchard, 2018.

    https://papers.nips.cc/paper/2018/hash/82f2b308c3b01637c607ce05f52a2fed-Abstract.html
    """

    grad_width: float = 0.5
    grad_scale: float = 1.0

    def __call__(self, v_mem, spike_threshold):
        abs_width = spike_threshold * self.grad_width
        return (
            self.grad_scale
            / abs_width
            * torch.exp(-torch.abs(v_mem - spike_threshold) / abs_width)
        )


@dataclass
class PeriodicExponential:
    """Surrogate gradient as defined in Weidel and Sheik, 2021.

    https://arxiv.org/abs/2111.01456
    """

    grad_width: float = 0.5
    grad_scale: float = 1.0

    def __call__(self, v_mem, spike_threshold):
        vmem_shifted = v_mem - spike_threshold / 2
        vmem_periodic = vmem_shifted - torch.div(
            vmem_shifted, spike_threshold, rounding_mode="floor"
        )
        vmem_below = vmem_shifted * (v_mem < spike_threshold)
        vmem_above = vmem_periodic * (v_mem >= spike_threshold)
        vmem_new = vmem_above + vmem_below
        spikePdf = (
            torch.exp(-torch.abs(vmem_new - spike_threshold / 2) / self.grad_width)
            / spike_threshold
        )

        return self.grad_scale * spikePdf
