from torch import nn
import torch


def input_diff_hook(self, input_, output):
    if len(input_) != 1:
        raise ValueError("Multiple inputs not supported for `input_diff_hook`")
    input_ = input_[0]

    # Difference between absolute output and output with absolute weights
    if isinstance(self, nn.Conv2d):
        abs_weight_output = torch.nn.functional.conv2d(
            input=input_,
            weight=torch.abs(self.weight),
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )
    else:
        abs_weight_output = nn.functional.linear(
            input=input_,
            weight=torch.abs(self.weight),
        )
    self.diff_output = abs_weight_output - torch.abs(output)

