from torch import nn
import sinabs.layers as sil
from warnings import warn
from sinabs import Network


def from_model(model, input_shape, input_conversion_layer=False,
               threshold=1.0, threshold_low=-1.0, membrane_subtract=1.0,
               exclude_negative_spikes=False):
    """
    Converts a Torch model and returns a Sinabs network object.
    Only sequential models or module lists are supported, with unpredictable
    behaviour on non-sequential models. This feature currently has limited
    capability. Supported layers are: Conv2d, AvgPool2d, MaxPool2d, Linear,
    BatchNorm2d (only if just after Linear or Conv2d), ReLU, Flatten,
    ZeroPad2d. LeakyReLUs are turned into ReLUs. Non-native torch layers
    supported are QuantizeLayer, YOLOLayer, NeuromorphicReLU, and
    DynapSumPoolLayer.

    :param model: a Torch model
    :param input_shape: the shape of the expected input
    :param input_conversion_layer: a Sinabs layer to be appended at the \
    beginning of the resulting network (typically Img2SpikeLayer or similar)
    :param threshold: The membrane potential threshold for spiking in \
    convolutional and linear layers (same for all layers).
    :param threshold_low: The lower bound of the potential in \
    convolutional and linear layers (same for all layers).
    :param membrane_subtract: Value subtracted from the potential upon \
    spiking for convolutional and linear layers (same for all layers).
    :return: :class:`.network.Network`
    """
    return SpkConverter(
        model,
        input_shape,
        input_conversion_layer,
        threshold,
        threshold_low,
        membrane_subtract,
        exclude_negative_spikes,
    ).convert()


class SpkConverter(object):
    def __init__(self, model, input_shape, input_conversion_layer=False,
                 threshold=1.0, threshold_low=-1.0, membrane_subtract=1.0,
                 exclude_negative_spikes=False):
        """
        Converts a Torch model and returns a Sinabs network object.
        Only sequential models or module lists are supported, with unpredictable
        behaviour on non-sequential models. This feature currently has limited
        capability. Supported layers are: Conv2d, AvgPool2d, MaxPool2d, Linear,
        BatchNorm2d (only if just after Linear or Conv2d), ReLU, Flatten,
        ZeroPad2d. LeakyReLUs are turned into ReLUs. Non-native torch layers
        supported are QuantizeLayer, YOLOLayer, NeuromorphicReLU, and
        DynapSumPoolLayer.

        :param model: a Torch model
        :param input_shape: the shape of the expected input
        :param input_conversion_layer: a Sinabs layer to be appended at the \
        beginning of the resulting network (typically Img2SpikeLayer or similar)
        :param threshold: The membrane potential threshold for spiking in \
        convolutional and linear layers (same for all layers).
        :param threshold_low: The lower bound of the potential in \
        convolutional and linear layers (same for all layers).
        :param membrane_subtract: Value subtracted from the potential upon \
        spiking for convolutional and linear layers (same for all layers).
        """
        self.model = model
        self.spk_mod = nn.Sequential()
        self.previous_layer_shape = input_shape
        self.index = 0
        self.leftover_rescaling = False
        self.threshold_low = threshold_low
        self.threshold = threshold
        self.membrane_subtract = membrane_subtract
        self.exclude_negative_spikes = exclude_negative_spikes

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
        print(name, self.previous_layer_shape)
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
            image_shape=self.previous_layer_shape[1:],
            kernel_shape=conv.kernel_size,
            channels_out=conv.out_channels,
            threshold=self.threshold,
            threshold_low=self.threshold_low,
            membrane_subtract=self.membrane_subtract,
            padding=(pad0, pad0, pad1, pad1),
            strides=conv.stride,
            bias=conv.bias is not None,
            negative_spikes=not self.exclude_negative_spikes,
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
        if not hasattr(pool.kernel_size, "__len__"):
            kernel = (pool.kernel_size, pool.kernel_size)
        else:
            kernel = pool.kernel_size
        if not hasattr(pool.stride, "__len__"):
            stride = (pool.stride, pool.stride)
        else:
            stride = pool.stride

        layer = sil.SumPooling2dLayer(
            pool_size=kernel,
            strides=stride,
            padding=(pool.padding, 0, pool.padding, 0),
            image_shape=self.previous_layer_shape[1:]
        )
        self.leftover_rescaling = 1 / (kernel[0] * kernel[1])
        self.add(f"avgpool_{self.index}", layer)

    def convert_maxpool2d(self, pool):
        """
        Converts a torch.nn.MaxPool2d layer to spiking and adds it to the
        spiking model.

        :param pool: the Torch layer to convert.
        """
        if not hasattr(pool.kernel_size, "__len__"):
            kernel = (pool.kernel_size, pool.kernel_size)
        else:
            kernel = pool.kernel_size
        if not hasattr(pool.stride, "__len__"):
            stride = (pool.stride, pool.stride)
        else:
            stride = pool.stride

        layer = sil.SpikingMaxPooling2dLayer(
            pool_size=kernel,
            strides=stride,
            padding=(pool.padding, 0, pool.padding, 0),
            image_shape=self.previous_layer_shape[1:]
        )
        self.add(f"maxpool_{self.index}", layer)

    def convert_sumpool(self, pool):
        """
        Converts a torch.nn.AvgPool2d layer to spiking and adds it to the
        spiking model.

        :param pool: the Torch layer to convert.
        """
        if not hasattr(pool.kernel_size, "__len__"):
            kernel = (pool.kernel_size, pool.kernel_size)
        else:
            kernel = pool.kernel_size
        if not hasattr(pool.stride, "__len__"):
            stride = (pool.stride, pool.stride)
        else:
            stride = pool.stride

        layer = sil.SumPooling2dLayer(
            pool_size=kernel,
            strides=stride,
            padding=(pool.padding, 0, pool.padding, 0),
            image_shape=self.previous_layer_shape[1:]
        )
        self.add(f"sumpool_{self.index}", layer)

    def convert_linear(self, lin):
        layer = sil.SpikingLinearLayer(
            in_features=lin.in_features,
            out_features=lin.out_features,
            threshold=self.threshold,
            threshold_low=self.threshold_low,
            membrane_subtract=self.membrane_subtract,
            bias=lin.bias is not None,
            negative_spikes=not self.exclude_negative_spikes,
        )

        if lin.bias is not None:
            layer.linear.bias.data = lin.bias.data.clone().detach()
        if self.leftover_rescaling:
            layer.linear.weight.data = (
                lin.weight * self.leftover_rescaling).clone().detach()
            self.leftover_rescaling = False
        else:
            layer.linear.weight.data = lin.weight.data.clone().detach()

        self.add(f"linear_{self.index}", layer)

    def previous_weighted_layer(self):
        """
        Identifies the previous convolution in the spiking model.
        Used to update convolution weights due to batch norm or avg pool.
        """
        last_layer = self.spk_mod._modules[list(self.spk_mod._modules)[-1]]
        if not isinstance(last_layer, (sil.SpikingConv2dLayer,
                                       sil.SpikingLinearLayer)):
            raise NotImplementedError(
                "Can convert this layer only after a convolution or linear layer.")
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

        last_convo = self.previous_weighted_layer()
        c_weight = last_convo.conv.weight.data.clone().detach()  # TODO this will give an error after Linear
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
            input_shape=self.previous_layer_shape[1:],
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
        self.previous_weighted_layer().negative_spikes = False

    def convert_zeropad2d(self, padlayer):
        """
        Converts a torch.nn.ZeroPad2d layer to spiking and adds it to the
        spiking model.

        :param padlayer: the Torch layer to convert.
        """
        layer = sil.ZeroPad2dLayer(
            image_shape=self.previous_layer_shape[1:],
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
            elif type(module).__name__ == "DynapSumPoolLayer":
                self.convert_sumpool(module)
            elif isinstance(module, nn.AvgPool2d):
                self.convert_avgpool(module)
            elif isinstance(module, nn.MaxPool2d):
                self.convert_maxpool2d(module)
            elif isinstance(module, nn.Linear):
                self.convert_linear(module)
            elif isinstance(module, nn.BatchNorm2d):
                self.convert_batchnorm(module)
            elif isinstance(module, nn.ReLU):
                self.convert_relu(module)
            elif isinstance(module, sil.NeuromorphicReLU):
                self.convert_relu(module)
            elif isinstance(module, sil.QuantizeLayer):
                pass
            elif isinstance(module, nn.LeakyReLU):
                self.convert_relu(module)
                warn("Leaky ReLU not supported. Converted to ReLU.")
            elif isinstance(module, nn.Flatten):
                self.add("flatten", sil.FlattenLayer(self.previous_layer_shape))
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
