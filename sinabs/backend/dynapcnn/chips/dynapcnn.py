from abc import abstractmethod
from typing import List
from warnings import warn

import samna
import torch
import sinabs

from sinabs.backend.dynapcnn.config_builder import ConfigBuilder
from sinabs.backend.dynapcnn.dvs_layer import DVSLayer, expand_to_pair
from sinabs.backend.dynapcnn.dynapcnn_layer import DynapcnnLayer
from sinabs.backend.dynapcnn.mapping import LayerConstraints


class DynapcnnConfigBuilder(ConfigBuilder):
    @classmethod
    @abstractmethod
    def get_samna_module(cls):
        """
        Get the samna parent module that hosts all the appropriate sub-modules and classes.

        Returns
        -------
        samna module
        """

    @classmethod
    @abstractmethod
    def get_default_config(cls):
        """
        Returns the default configuration for the device type
        """

    @classmethod
    def write_dvs_layer_config(cls, layer: DVSLayer, config: "DvsLayerConfig"):
        for param, value in layer.get_config_dict().items():
            setattr(config, param, value)

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
        dimensions["padding"] = {
            "x": layer.conv_layer.padding[1],
            "y": layer.conv_layer.padding[0],
        }
        dimensions["stride"] = {
            "x": layer.conv_layer.stride[1],
            "y": layer.conv_layer.stride[0],
        }
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
            raise Exception(
                "Unknown reset mechanism. Only MembraneReset and MembraneSubtract are currently understood."
            )

        # if (not return_to_zero) and self.spk_layer.membrane_subtract != self.spk_layer.threshold:
        #    warn(
        #        "SpikingConv2dLayer: Subtraction of membrane potential is always by high threshold."
        #    )
        if layer.spk_layer.min_v_mem is None:
            min_v_mem = -(2**15)
        else:
            min_v_mem = int(layer.spk_layer.min_v_mem)
        config_dict.update(
            {
                "return_to_zero": return_to_zero,
                "threshold_high": int(layer.spk_layer.spike_threshold),
                "threshold_low": min_v_mem,
                "monitor_enable": False,
                "neurons_initial_value": neurons_state.int().tolist(),
            }
        )
        # Update parameters from pooling
        if layer.pool_layer is not None:
            config_dict["destinations"][0]["pooling"] = expand_to_pair(
                layer.pool_layer.kernel_size
            )[0]
            config_dict["destinations"][0]["enable"] = True
        else:
            pass

        return config_dict

    @classmethod
    def write_dynapcnn_layer_config(
        cls, layer: DynapcnnLayer, chip_layer: "CNNLayerConfig"
    ):
        """Write a single layer configuration to the dynapcnn conf object.

        Parameters
        ----------
            layer:
                The dynapcnn layer to write the configuration for
            chip_layer: CNNLayerConfig
                DYNAPCNN configuration object representing the layer to which
                configuration is written.
        """
        config_dict = cls.get_dynapcnn_layer_config_dict(layer=layer)
        # Update configuration of the DYNAPCNN layer
        chip_layer.dimensions = config_dict["dimensions"]
        config_dict.pop("dimensions")
        for i in range(len(config_dict["destinations"])):
            if "pooling" in config_dict["destinations"][i]:
                chip_layer.destinations[i].pooling = config_dict["destinations"][i][
                    "pooling"
                ]
        config_dict.pop("destinations")
        for param, value in config_dict.items():
            try:
                setattr(chip_layer, param, value)
            except TypeError as e:
                raise TypeError(f"Unexpected parameter {param} or value. {e}")

    @classmethod
    def build_config(cls, model: "DynapcnnNetwork", chip_layers: List[int]):
        layers = model.sequence
        config = cls.get_default_config()

        has_dvs_layer = False
        i_cnn_layer = 0  # Instantiate an iterator for the cnn cores
        for i, chip_equivalent_layer in enumerate(layers):
            if isinstance(chip_equivalent_layer, DVSLayer):
                chip_layer = config.dvs_layer
                cls.write_dvs_layer_config(chip_equivalent_layer, chip_layer)
                has_dvs_layer = True
            elif isinstance(chip_equivalent_layer, DynapcnnLayer):
                chip_layer = config.cnn_layers[chip_layers[i_cnn_layer]]
                cls.write_dynapcnn_layer_config(chip_equivalent_layer, chip_layer)
                i_cnn_layer += 1
            else:
                # in our generated network there is a spurious layer...
                # should never happen
                raise TypeError("Unexpected layer in the model")

            if i == len(layers) - 1:
                # last layer
                chip_layer.destinations[0].enable = False
            else:
                # Set destination layer
                chip_layer.destinations[0].layer = chip_layers[i_cnn_layer]
                chip_layer.destinations[0].enable = True

        if not has_dvs_layer:
            config.dvs_layer.pass_sensor_events = False

        return config

    @classmethod
    def get_constraints(cls) -> List[LayerConstraints]:
        ## Chip specific constraints
        weight_memory_size = [
            16 * 1024,  # 0
            16 * 1024,  # 1
            16 * 1024,  # 2
            32 * 1024,  # 3
            32 * 1024,  # 4
            64 * 1024,  # 5
            64 * 1024,  # 6
            16 * 1024,  # 7
            16 * 1024,
        ]  # _WEIGHTS_MEMORY_SIZE

        neurons_memory_size = [
            64 * 1024,  # 0
            64 * 1024,  # 1
            64 * 1024,  # 2
            32 * 1024,  # 3
            32 * 1024,  # 4
            16 * 1024,  # 5
            16 * 1024,  # 6
            16 * 1024,  # 7
            16 * 1024,
        ]  # 8
        bias_memory_size = [1024] * 9

        constraints = [
            LayerConstraints(km, nm, bm)
            for (km, nm, bm) in zip(
                weight_memory_size, neurons_memory_size, bias_memory_size
            )
        ]
        return constraints

    @classmethod
    def monitor_layers(cls, config, layers: List):
        """Updates the config object in place.

        Parameters
        ----------
        config:
            samna config object
        monitor_chip_layers:
            The layers to be monitored on the chip.

        Returns
        -------
        config:
            Returns the modified config. (The config object is modified in place)
        """
        monitor_layers = layers.copy()
        if "dvs" in monitor_layers:
            config.dvs_layer.monitor_enable = True
            if config.dvs_layer.pooling.x != 1 or config.dvs_layer.pooling.y != 1:
                warn(
                    f"DVS layer has pooling and is being monitored. "
                    "Note that pooling will not be reflected in the monitored events."
                )
            monitor_layers.remove("dvs")
        for lyr_indx in monitor_layers:
            config.cnn_layers[lyr_indx].monitor_enable = True
            if any(
                dest.pooling != 1 for dest in config.cnn_layers[lyr_indx].destinations
            ):
                warn(
                    f"Layer {lyr_indx} has pooling and is being monitored. "
                    "Note that pooling will not be reflected in the monitored events."
                )
        return config

    @classmethod
    @abstractmethod
    def get_input_buffer(cls):
        """
        Initialize and return the appropriate output buffer object Note that this just the
        buffer object.

        This does not actually connect the buffer object to the graph. (It is needed as of samna
        0.21.0)
        """

    @classmethod
    @abstractmethod
    def get_output_buffer(cls):
        """
        Initialize and return the appropriate output buffer object Note that this just the
        buffer object.

        This does not actually connect the buffer object to the graph.
        """

    @classmethod
    def reset_states(cls, config, randomize=False):
        """
        Parameters
        ----------
        config:
            samna config object
        randomize (bool):
            If true, the states will be set to random initial values. Else, they will be set to zero
        """
        for idx, lyr in enumerate(config.cnn_layers):
            shape = torch.tensor(lyr.neurons_initial_value).shape
            # set the config's neuron initial state values into zeros
            if randomize:
                new_state = torch.randint(
                    lyr.threshold_low, lyr.threshold_high, shape
                ).tolist()
            else:
                new_state = torch.zeros(shape, dtype=torch.int).tolist()
            config.cnn_layers[idx].neurons_initial_value = new_state
        return config
