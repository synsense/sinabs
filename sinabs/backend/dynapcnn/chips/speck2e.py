import torch
from subprocess import CalledProcessError
from sinabs.backend.dynapcnn.dynapcnn_layer import DynapcnnLayer
try:
    import samna
    from samna.speck2e.configuration import SpeckConfiguration
except (ImportError, ModuleNotFoundError, CalledProcessError):
    SAMNA_AVAILABLE = False
else:
    SAMNA_AVAILABLE = True
from .dynapcnn import DynapcnnConfigBuilder
from sinabs.backend.dynapcnn.dvs_layer import expand_to_pair


# Since most of the configuration is identical to DYNAP-CNN, we can simply inherit this class

class Speck2EConfigBuilder(DynapcnnConfigBuilder):

    @classmethod
    def get_samna_module(cls):
        return samna.speck2e

    @classmethod
    def get_default_config(cls) -> "SpeckConfiguration":
        return SpeckConfiguration()

    @classmethod
    def get_output_buffer(cls):
        return samna.BasicSinkNode_speck2e_event_output_event()

    
    @classmethod
    def get_dynapcnn_layer_config_dict(cls, layer: DynapcnnLayer):
        config_dict = {}
        config_dict["destinations"] = [{}, {}]

        # Update the dimensions
        channel_count, input_size_y, input_size_x = layer.input_shape
        dimensions = {"input_shape": {}, "output_shape": {}}
        dimensions["input_shape"]["size"] = {"x": input_size_x, "y": input_size_y}
        dimensions["input_shape"]["feature_count"] = channel_count

        # dimensions["output_feature_count"] already done in conv2d_to_dict
        (f, h, w) = layer.get_neuron_shape()
        dimensions["output_shape"]["size"] = {}
        dimensions["output_shape"]["feature_count"] = f
        dimensions["output_shape"]["size"]["x"] = w
        dimensions["output_shape"]["size"]["y"] = h
        dimensions["padding"] = {"x": layer.conv_layer.padding[1], "y": layer.conv_layer.padding[0]}
        dimensions["stride"] = {"x": layer.conv_layer.stride[1], "y": layer.conv_layer.stride[0]}
        dimensions["kernel_size"] = layer.conv_layer.kernel_size[0]

        if dimensions["kernel_size"] != layer.conv_layer.kernel_size[1]:
            raise ValueError("Conv2d: Kernel must have same height and width.")
        config_dict["dimensions"] = dimensions
        # Update parameters from convolution
        if layer.conv_layer.bias is not None:
            (weights, biases) = layer.conv_layer.parameters()
        else:
            (weights,) = layer.conv_layer.parameters()
            biases = torch.zeros(layer.conv_layer.out_channels)
        weights = weights.transpose(2, 3)  # Need this to match samna convention
        config_dict["weights"] = weights.int().tolist()
        config_dict["biases"] = biases.int().tolist()
        config_dict["leak_enable"] = biases.bool().any()

        # Update parameters from the spiking layer

        # - Neuron states
        if not layer.spk_layer.is_state_initialised():
            # then we assign no initial neuron state to DYNAP-CNN.
            f, h, w = layer.get_neuron_shape()
            neurons_state = torch.zeros(f, w, h)
        elif layer.spk_layer.v_mem.dim() == 4:
            # 4-dimensional states should be the norm when there is a batch dim
            neurons_state = layer.spk_layer.v_mem.transpose(2, 3)[0]
        else:
            raise ValueError(
                f"Current v_mem (shape: {layer.spk_layer.v_mem.shape}) of spiking layer not understood."
            )

        # - Resetting vs returning to 0
        if isinstance(layer.spk_layer.reset_fn, sinabs.activation.MembraneReset):
            return_to_zero = True
        elif isinstance(layer.spk_layer.reset_fn, sinabs.activation.MembraneSubtract):
            return_to_zero = False
        else:
            raise Exception("Unknown reset mechanism. Only MembraneReset and MembraneSubtract are currently understood.")

        #if (not return_to_zero) and self.spk_layer.membrane_subtract != self.spk_layer.threshold:
        #    warn(
        #        "SpikingConv2dLayer: Subtraction of membrane potential is always by high threshold."
        #    )
        if layer.spk_layer.min_v_mem is None:
            min_v_mem = -2**15
        else:
            min_v_mem = int(layer.spk_layer.min_v_mem)
        config_dict.update({
            "return_to_zero": return_to_zero,
            "threshold_high": int(layer.spk_layer.spike_threshold),
            "threshold_low": min_v_mem,
            "monitor_enable": False,
            "neurons_initial_value": neurons_state.int().tolist(),
        })
        # Update parameters from pooling
        if layer.pool_layer is not None:
            config_dict["destinations"][0]["pooling"] = expand_to_pair(layer.pool_layer.kernel_size)[0]
            config_dict["destinations"][0]["enable"] = True
        else:
            pass


        return config_dict