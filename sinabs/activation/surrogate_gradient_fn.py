from dataclasses import dataclass, field


@dataclass
class Heaviside:
    window: float = 1.

    def __call__(self, v_mem, threshold):
        return ((v_mem >= (threshold - self.window)).float()) / threshold 
