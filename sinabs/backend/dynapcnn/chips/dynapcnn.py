import copy
from typing import List, Union, Dict
from warnings import warn

import samna
import torch
from samna.dynapcnn.configuration import DynapcnnConfiguration

import sinabs
from sinabs.backend.dynapcnn.config_builder import ConfigBuilder
from sinabs.backend.dynapcnn.dvs_layer import DVSLayer, expand_to_pair
from sinabs.backend.dynapcnn.dynapcnn_layer import DynapcnnLayer
from sinabs.backend.dynapcnn.mapping import LayerConstraints

import sinabs

class DynapcnnConfigBuilder(ConfigBuilder):
    @classmethod
    def get_samna_module(cls):
        return samna.dynapcnn

    @classmethod
    def get_default_config(cls) -> "DynapcnnConfiguration":
        return DynapcnnConfiguration()

    @classmethod
    def get_dvs_layer_config_dict(cls, layer: DVSLayer): ...

    @classmethod
    def write_dvs_layer_config(cls, layer: DVSLayer, config: "DvsLayerConfig"):
        for param, value in layer.get_config_dict().items():
            setattr(config, param, value)

    @classmethod
    def set_kill_bits(cls, layer: DynapcnnLayer, config_dict: dict) -> dict:
        """This method updates all the kill_bit parameters.

        Args:
            layer (DynapcnnLayer): The layer of whome the configuration is to be generated
            config_dict (dict): The dictionary where the parameters need to be added


        Returns:
            dict: returns the updated config_dict.
        """
        config_dict = copy.deepcopy(config_dict)

        if layer.conv_layer.bias is not None:
            (weights, biases) = layer.conv_layer.parameters()
        else:
            (weights,) = layer.conv_layer.parameters()
            biases = torch.zeros(layer.conv_layer.out_channels)

        config_dict["weights_kill_bit"] = (~weights.bool()).tolist()
        config_dict["biases_kill_bit"] = (~biases.bool()).tolist()

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

        config_dict["neurons_value_kill_bit"] = (
            torch.zeros_like(neurons_state).bool().tolist()
        )

        return config_dict

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
        # config_dict["weights_kill_bit"] = torch.zeros_like(weights).bool().tolist()
        # config_dict["biases_kill_bit"] = torch.zeros_like(biases).bool().tolist()

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
                # "neurons_value_kill_bit" : torch.zeros_like(neurons_state).bool().tolist()
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

        # Set kill bits
        config_dict = cls.set_kill_bits(layer=layer, config_dict=config_dict)

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
    def write_dynapcnn_layer_config_graph(cls, dcnnl: DynapcnnLayer, chip_layer: "CNNLayerConfig", forward_map: dict) -> None:
        """ Uses the data in `dcnnl` to configure a `CNNLayerConfig` to be deployed on chip.

        Parameters
        ----------
            dcnnl (DynapcnnLayer): the layer for which the condiguration will be written.
            chip_layer (CNNLayerConfig): used to represent/configure `dcnnl` onto the chip.
            forward_map (dict): a dictionary with keys being the ID of each DynapcnnLayer and values being the layer 
                itself. This is used to retrieve the `.assigned_core` for each of the layers in `.dynapcnnlayer_destination` 
                such that `chip_layer.destinations` can be configured.
        """

        # extracting from a DynapcnnLayer the config. variables for its CNNLayerConfig.
        config_dict = dcnnl.get_layer_config_dict()

        # use core indexing instead of DynapcnnLayer indexing for destinations.
        for dest_config in config_dict['destinations']:
            dcnnl_idx = dest_config['layer']

            # get the core the destination DynapcnnLayer is using.
            dcnnl_core_idx = forward_map[dcnnl_idx].assigned_core

            dest_config['layer'] = dcnnl_core_idx

        # set the destinations configuration.
        for i in range(len(config_dict['destinations'])):
            chip_layer.destinations[i] = config_dict['destinations'][i]
        config_dict.pop("destinations")

        # set remaining configuration.
        for param, value in config_dict.items():                                                      # set remaining attributes.
            try:
                setattr(chip_layer, param, value)
            except TypeError as e:
                raise TypeError(f"Unexpected parameter {param} or value. {e}")

    @classmethod
    def build_config(cls, model: Union["DynapcnnNetwork"], chip_layers: Union[List[int], None]) -> DynapcnnConfiguration:
        """ Uses `DynapcnnLayer` objects to configure their equivalent chip core via a `CNNLayerConfig` object that is built
        using using the `DynapcnnLayer` properties. 

        Parameters
        ----------
            model:
                either a `DynapcnnNetwork` or a `DynapcnnNetworkGraph` instance where the model (DynapcnnLayer) layers can be found.
            chip_layers:
                a list containing the core indexes where each `DynapcnnLayer` will be mapped to (if `model` is an instance of `DynapcnnNetwork`, otherwise `None`).

        Returns
        ----------
            config:
                an instance of a `DynapcnnConfiguration`.
        """
        config = cls.get_default_config()

        if type(model) == sinabs.backend.dynapcnn.dynapcnn_network.DynapcnnNetwork:
            """ Loops through `DynapcnnNetworkGraph._forward_map`, containing all `DynapcnnLayer`s in the model, their
            core ID to be loaded onto and their target destinations. Each `ith_dcnnl` has all the info. necessary to config.
            their respective `CNNLayerConfig` object.
            """
            has_dvs_layer = False   # TODO DVSLayer not supported yet.

            for layer_index, ith_dcnnl in model.forward_map.items():
                if isinstance(ith_dcnnl, DVSLayer):
                    # TODO DVSLayer not supported yet.
                    pass

                elif isinstance(ith_dcnnl, DynapcnnLayer):
                    chip_layer = config.cnn_layers[ith_dcnnl.assigned_core]
                    cls.write_dynapcnn_layer_config_graph(ith_dcnnl, chip_layer, model.forward_map)

                else:
                    # shouldn't happen since type checks are made previously.
                    raise TypeError(f"Layer (index {layer_index}) is unexpected in the model: \n{ith_dcnnl}")
                
            if not has_dvs_layer:
                # TODO DVSLayer not supported yet.
                config.dvs_layer.pass_sensor_events = False
            else:
                config.dvs_layer.pass_sensor_events = False

        else:
            raise TypeError(f"Unexpected model {type(model)}.")

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
    def monitor_layers(cls, config: "DynapcnnConfiguration", layers: List):
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
    def get_input_buffer(cls):
        return samna.BasicSourceNode_dynapcnn_event_input_event()

    @classmethod
    def get_output_buffer(cls):
        return samna.BasicSinkNode_dynapcnn_event_output_event()

    @classmethod
    def reset_states(cls, config: DynapcnnConfiguration, randomize=False):
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
