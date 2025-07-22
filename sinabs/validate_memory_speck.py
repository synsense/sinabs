from typing import Tuple
from matplotlib import pyplot as plt
from matplotlib import colors

import numpy as np


class ValidateMapping:
    def __init__(
        self,
        input_feature_size: int,
        output_feature_size: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int],
        input_dimension: Tuple[int, int] = [64, 64],
        conv_2d: bool = True,
    ):
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_dimension = input_dimension

        # - If not Conv2D layer, assuming it is AvgPool2D layer
        if not conv_2d:
            if (
                kernel_size[0] != kernel_size[1]
                or kernel_size[0] == 3
                or kernel_size[0] > 4
            ):
                raise Exception(
                    "Kernel size is limited to 1x1, 2x2 or 4x4 for AvgPool2D layer."
                )

        if (
            len(kernel_size) > 2
            or len(stride) > 2
            or len(padding) > 2
            or len(input_dimension) > 2
        ):
            raise Exception(
                "We expect input dimension kernel, stride and padding to be 2D elements, i.e.,"
                "to have only two positions: x and y."
            )

        if kernel_size[0] > 16 or kernel_size[1] > 16:
            raise Exception("Kernel size is limited to, at most, 16x16.")

        if output_feature_size > 1024:
            raise Exception("Output feature size is limited to, at most, 1024.")

        if not self.check_stride():
            raise Warning("Kernel stride can be 1, 2, 4 or 8 and, at most, 8x8.")

    def calculate_total_memory(self):
        kernel_memory = self.calculate_kernel_memory()
        neuron_memory = self.calculate_neuron_memory()

        kernel_error_msg = self.verify_combined_memories(
            "kernel", kernel_memory, "neuron", neuron_memory
        )
        neuron_error_msg = self.verify_combined_memories(
            "neuron", neuron_memory, "kernel", kernel_memory
        )

        print(kernel_error_msg)
        print(neuron_error_msg)
        return (
            kernel_memory / 1024,
            neuron_memory / 1024,
            kernel_error_msg,
            neuron_error_msg,
        )

    def calculate_kernel_memory(self):
        return self.input_feature_size * pow(
            2,
            np.ceil(np.log2(self.kernel_size[0] * self.kernel_size[1]))
            + np.ceil(np.log2(self.output_feature_size)),
        )

    def calculate_neuron_memory(self):
        fx = (
            (self.input_dimension[0] - self.kernel_size[0] + 2 * self.padding[0])
            / self.stride[0]
        ) + 1
        fy = (
            (self.input_dimension[1] - self.kernel_size[1] + 2 * self.padding[1])
            / self.stride[1]
        ) + 1
        return self.output_feature_size * pow(
            2, (np.ceil(np.log2(fx)) + np.ceil(np.log2(fy)))
        )

    def check_stride(self):
        for i in range(len(self.stride)):
            if (
                self.stride[i] == 1
                or self.stride[i] == 2
                or self.stride[i] == 4
                or self.stride[i] == 8
            ):
                return True
        return False

    def verify_combined_memories(
        self, base_name: str, base_memory: int, compared_name: str, compared_memory: int
    ):
        # core ids --------- kernel memory -------- neuron memory
        # [0, 1, 2] ------------- 16Ki ----------------- 64Ki
        # [3, 4] ---------------- 32Ki ----------------- 32Ki
        # [5, 6] ---------------- 64Ki ----------------- 16Ki
        # [7, 8] ---------------- 16Ki ----------------- 16Ki

        base_memory = base_memory / 1024
        compared_memory = compared_memory / 1024

        error_msg = ""
        if base_memory > 64:
            error_msg = (
                f"{base_name.capitalize()} memory is {base_memory:g}Ki and can not be mapped on chip. "
                f"{base_name.capitalize()} memory on chip needs to be at most 64Ki."
            )

        if base_memory > 16 and base_memory <= 32:
            if compared_memory > 32:
                error_msg = (
                    "There is no core on chip to fit neuron and kernel memories. "
                    f"When {base_name} memory is higher than 16Ki, {compared_name} memory needs to be at most 32Ki. "
                    f"{base_name.capitalize()} is {base_memory:g}Ki and {compared_name} is {compared_memory:g}Ki."
                )

        if base_memory > 32:
            if compared_memory > 16:
                error_msg = (
                    "There is no core on chip to fit neuron and kernel memories. "
                    f"When {base_name} memory is higher than 32Ki, {compared_name} memory needs to be at most 16Ki. "
                    f"{base_name.capitalize()} is {base_memory:g}Ki and {compared_name} is {compared_memory:g}Ki."
                )

        return error_msg
