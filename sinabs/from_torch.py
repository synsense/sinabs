from torch import nn
import sinabs.layers as sil
from warnings import warn
from sinabs import Network


def from_model(model, input_shape, input_conversion_layer=False,
               conv_threshold_low=None):
    """
    Converts a Torch model and returns a Sinabs network object.
    Only sequential models or module lists are supported, with unpredictable
    behaviour on non-sequential models. This feature currently has limited
    capability.

    :param model: a Torch model
    :param input_shape: the shape of the expected input
    :param input_conversion_layer: a Sinabs layer to be appended at the \
    beginning of the resulting network (typically Img2SpikeLayer or similar)
    :param conv_threshold_low: The lower bound of the potential in \
    convolutional layers (same for all layers).
    :return: :class:`.network.Network`
    """
    return SpkConverter(
        model,
        input_shape,
        input_conversion_layer,
        conv_threshold_low
    ).convert()


class SpkConverter(object):
    def __init__(self, model, input_shape, input_conversion_layer=False,
                 conv_threshold_low=None):
        """
        Converts a Torch model and returns a Sinabs network object.
        Only sequential models or module lists are supported, with unpredictable
        behaviour on non-sequential models. This feature currently has limited
        capability.

        :param model: a Torch model
        :param input_shape: the shape of the expected input
        :param input_conversion_layer: a Sinabs layer to be appended at the \
        beginning of the resulting network (typically Img2SpikeLayer or similar)
        :param conv_threshold_low: The lower bound of the potential in \
        convolutional layers (same for all layers).
        """
        self.model = model
        self.spk_mod = nn.Sequential()
        self.previous_layer_shape = input_shape
        self.index = 0
        self.leftover_rescaling = False
        self.conv_threshold_low = conv_threshold_low

        if input_conversion_layer:
            self.add("input_conversion", input_conversion_layer)

    def modules(self):
        """
        Lists all modules in the model, except the model itself,
        Sequentials and ModuleLists.
        """
        for mname, m in self.model.named_modules():
            if (isinstance(m, nn.Sequential)
                or isinstance(m, nn.ModuleList)
                or m.__class__ == self.model.__class__):
                continue
            yield (mname, m)

    def add(self, name, module):
        """
        Adds a layer to the spiking model.

        :param name: the name of the layer
        :param module: the layer to add
        """
        self.spk_mod.add_module(name, module)
        self.previous_layer_shape = module.get_output_shape(
            self.previous_layer_shape)
        print(self.previous_layer_shape)
        self.index += 1

    def convert_conv2d(self, conv):
        """
        Converts a torch.nn.Conv2d layer to spiking and adds it to the spiking
        model.

        :param conv: the Torch layer to convert.
        """
        pad0, pad1 = conv.padding
        layer = sil.SpikingConv2dLayer(
            channels_in=conv.in_channels,
            image_shape=self.previous_layer_shape,
            kernel_shape=conv.kernel_size,
            channels_out=conv.out_channels,
            padding=(pad0, pad0, pad1, pad1),
            strides=conv.stride,
            bias=conv.bias is not None,
            negative_spikes=True,
            threshold_low=self.conv_threshold_low,
        )

        if conv.bias is not None:
            layer.conv.bias.data = conv.bias.data.clone().detach()
        if self.leftover_rescaling:
            layer.conv.weight.data = (conv.weight *
                                      self.leftover_rescaling).clone().detach()
            self.leftover_rescaling = False
        else:
            layer.conv.weight.data = conv.weight.data.clone().detach()

        self.add(f"conv2d_{self.index}", layer)

    def convert_avgpool(self, pool):
        """
        Converts a torch.nn.AvgPool2d layer to spiking and adds it to the
        spiking model.

        :param pool: the Torch layer to convert.
        """
        layer = sil.SumPooling2dLayer(
            pool_size=(pool.kernel_size, pool.kernel_size),
            strides=(pool.stride, pool.stride),
            padding=(pool.padding, 0, pool.padding, 0),
            image_shape=self.previous_layer_shape
        )
        self.leftover_rescaling = 0.25
        self.add(f"avgpool_{self.index}", layer)

    def previous_convo(self):
        """
        Identifies the previous convolution in the spiking model.
        Used to update convolution weights due to batch norm or avg pool.
        """
        last_layer = self.spk_mod._modules[list(self.spk_mod._modules)[-1]]
        if not isinstance(last_layer, sil.SpikingConv2dLayer):
            raise NotImplementedError(
                "Can convert this layer only after a convolution.")
        return last_layer

    def convert_batchnorm(self, bn):
        """
        Converts a torch.nn.BatchNorm2d layer to spiking and adds it to the
        spiking model.

        :param bn: the Torch layer to convert.
        """
        mu = bn.running_mean
        sigmasq = bn.running_var

        if bn.affine:
            gamma, beta = bn.weight, bn.bias
        else:
            gamma, beta = 1.0, 0.0

        factor = gamma / sigmasq.sqrt()

        last_convo = self.previous_convo()
        c_weight = last_convo.conv.weight.data.clone().detach()
        c_bias = 0. if last_convo.conv.bias is None else last_convo.conv.bias.data.clone().detach()

        last_convo.conv.weight.data = c_weight * factor[:, None, None, None]
        last_convo.conv.bias = nn.Parameter((beta + (c_bias - mu) * factor))

    def convert_yolo(self, yolo):
        """
        This feature is experimental.

        Converts a YOLO layer to spiking and adds it to the
        spiking model. Note that the Sinabs YOLO layer converts
        spikes to rates and is not a spiking layer. YOLO layers
        differ in implementation, and this will work only for YOLO
        layers similar to the Sinabs YOLO layer.

        :param yolo: the YOLO layer to convert.
        """
        new_yolo = sil.YOLOLayer(
            anchors=yolo.anchors,
            num_classes=yolo.num_classes,
            img_dim=416,  # TODO
            return_loss=False,
            compute_rate=True
        )

        self.spk_mod.add_module(f"yolo_{self.index}", new_yolo)

    def convert_relu(self, relu):
        """
        Converts a torch.nn.ReLU layer to spiking, by preventing the
        previous convolutional layer from emitting negative spikes.

        :param relu: the Torch layer to convert.
        """
        self.previous_convo().negative_spikes = False

    def convert_zeropad2d(self, padlayer):
        """
        Converts a torch.nn.ZeroPad2d layer to spiking and adds it to the
        spiking model.

        :param padlayer: the Torch layer to convert.
        """
        layer = sil.ZeroPad2dLayer(
            image_shape=self.previous_layer_shape,
            padding=padlayer.padding
        )
        self.add("zeropad2d", layer)

    def convert(self):
        """
        Converts the Torch model and returns a Sinabs network object.

        :returns network: the Sinabs network object created by conversion.
        """
        for mname, module in self.modules():
            if isinstance(module, nn.Conv2d):
                self.convert_conv2d(module)
            elif isinstance(module, nn.AvgPool2d):
                self.convert_avgpool(module)
            elif isinstance(module, nn.BatchNorm2d):
                self.convert_batchnorm(module)
            elif isinstance(module, nn.ReLU):
                self.convert_relu(module)
            elif isinstance(module, nn.LeakyReLU):
                self.convert_relu(module)
                warn("Leaky ReLU not supported. Converted to ReLU.")
            elif isinstance(module, nn.ZeroPad2d):
                self.convert_zeropad2d(module)
            elif type(module).__name__ == "YOLOLayer":
                self.convert_yolo(module)
                break
            else:
                warn(f"Layer '{type(module).__name__}' is not supported. Skipping!")

        if self.leftover_rescaling:
            warn("Caution: the rescaling due to the last average pooling could not be applied!")

        network = Network()
        network.spiking_model = self.spk_mod
        network.analog_model = self.model

        return network
