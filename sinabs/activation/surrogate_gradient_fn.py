from dataclasses import dataclass
import math


@dataclass
class Heaviside:
    """
    Heaviside surrogat gradient with optional shift.
    
    Parameters:
        window:
            Distance between step of Heaviside surrogate gradient and 
            threshold, relative to threshold.
    """
    
    window: float = 1.

    def __call__(self, v_mem, threshold):
        return ((v_mem >= (threshold - self.window)).float()) / threshold 


@dataclass
class MultiGaussian:
    """
    Surrogate gradient as defined in Yin et al., 2021.
    https://www.biorxiv.org/content/10.1101/2021.03.22.436372v2
    """
    mu: float = 0.
    sigma: float = 0.5
    
    def __call__(self, v_mem, threshold):
        return torch.exp(-(((v_mem-threshold) - self.mu) ** 2) / (2 * self.sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / self.sigma

