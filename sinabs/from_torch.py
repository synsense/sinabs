from torch import nn
import sinabs.layers as sil
from warnings import warn


class ConvertedNet(nn.Module):
    def __init__(self, spk_mod):
        super().__init__()
        self.module_list = spk_mod

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
        return x


class SpkConverter(object):
    def __init__(self, model, input_shape, input_conversion_layer=False):
        self.model = model
        self.spk_mod = nn.ModuleList()
        self.previous_layer_shape = input_shape
        self.index = 0
        self.leftover_rescaling = False
        if input_conversion_layer:
            self.add("input_conversion", input_conversion_layer)

    def modules(self):
        for mname, m in self.model.named_modules():
            if (isinstance(m, nn.Sequential)
                or isinstance(m, nn.ModuleList)
                or m.__class__ == self.model.__class__):
                continue
            yield (mname, m)

    def add(self, name, module):
        self.spk_mod.add_module(name, module)
        self.previous_layer_shape = module.get_output_shape(
            self.previous_layer_shape)
        self.index += 1

    def convert_conv2d(self, conv):
        layer = sil.SpikingConv2dLayer(
                    channels_in=conv.in_channels,
                    image_shape=self.previous_layer_shape,
                    kernel_shape=conv.kernel_size,
                    channels_out=conv.out_channels,
                    padding=conv.padding,
                    strides=conv.stride,
                    bias=conv.bias is not None,
                    negative_spikes=True
        )

        layer.conv.bias = conv.bias
        if self.leftover_rescaling:
            layer.conv.weight = nn.Parameter(conv.weight *
                                             self.leftover_rescaling)
            self.leftover_rescaling = False
        else:
            layer.conv.weight = conv.weight

        self.add(f"conv2d_{self.index}", layer)

    def convert_avgpool(self, pool):
        layer = sil.SumPooling2dLayer(
            pool_size=(pool.kernel_size, pool.kernel_size),
            strides=(pool.stride, pool.stride),
            padding=(pool.padding, pool.padding, 0, 0),
            image_shape=self.previous_layer_shape
        )
        self.leftover_rescaling = 0.25
        self.add(f"avgpool_{self.index}", layer)

    def previous_convo(self):
        last_layer = self.spk_mod._modules[list(self.spk_mod._modules)[-1]]
        if not isinstance(last_layer, sil.SpikingConv2dLayer):
            raise NotImplementedError(
                "Can convert this layer only after a convolution.")
        return last_layer

    def convert_batchnorm(self, bn):
        mu = bn.running_mean
        sigmasq = bn.running_var

        if bn.affine:
            gamma, beta = bn.weight, bn.bias
        else:
            gamma, beta = 1.0, 0.0

        factor = gamma / sigmasq.sqrt()

        last_convo = self.previous_convo()
        c_weight = last_convo.conv.weight
        c_bias = 0. if last_convo.conv.bias is None else last_convo.conv.bias
        last_convo.conv.weight = nn.Parameter(c_weight *
                                              factor[:, None, None, None])
        last_convo.conv.bias = nn.Parameter((beta + (c_bias - mu) * factor))

    def convert_relu(self, relu):
        self.previous_convo().negative_spikes = False

    def convert(self):
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
            else:
                warn(f"Layer '{module.__class__}' is not supported. Skipping!")

        if self.leftover_rescaling:
            warn("Caution: the rescaling due to the last average pooling could not be applied!")

        return ConvertedNet(self.spk_mod)
