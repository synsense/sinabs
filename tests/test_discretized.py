from backend.Speck.discretize import discretize_sl
import numpy as np
import torch
from sinabs.from_torch import from_model

threshold = np.random.random() * 10
thr_low = np.random.random() * 10

weights = torch.tensor(np.random.normal(0, 10, size=(4, 1, 3, 3)))
inp = torch.tensor(np.random.choice(3, size=(30, 1, 32, 32))).float()

module = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3)),
)

module_spk = from_model(module, input_shape=(1, 32, 32), threshold=threshold, threshold_low=-thr_low)
module_discr = discretize_sl(module_spk, to_int=False)

print(module_spk(inp).sum().item())
print(module_discr(inp).sum().item())
