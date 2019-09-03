import torch
from torch import nn
import sinabs.layers as sil
from warnings import warn
from copy import deepcopy


class ConvertedNet(nn.Module):
    def __init__(self, spk_mod):
        super().__init__()
        self.module_list = spk_mod

    def forward(self, x):
        for m in self.module_list:
            x = m(x)
        return x


class Spk2Rates(sil.TorchLayer):
    def __init__(self, input_shape=None, layer_name="spk2rates"):
        sil.TorchLayer.__init__(
            self, input_shape=input_shape, layer_name=layer_name
        )

    def forward(self, x):
        return x.float().mean(0).unsqueeze(0)

    def get_output_shape(self, input_shape):
        return input_shape


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
        print(self.previous_layer_shape)
        self.index += 1

    def convert_conv2d(self, conv):
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
            threshold_low=None,
        )

        if conv.bias is not None:
            layer.conv.bias.data = torch.tensor(conv.bias.data)
        if self.leftover_rescaling:
            layer.conv.weight.data = torch.tensor(conv.weight *
                                                  self.leftover_rescaling)
            self.leftover_rescaling = False
        else:
            layer.conv.weight.data = torch.tensor(conv.weight.data)

        self.add(f"conv2d_{self.index}", layer)

    def convert_avgpool(self, pool):
        layer = sil.SumPooling2dLayer(
            pool_size=(pool.kernel_size, pool.kernel_size),
            strides=(pool.stride, pool.stride),
            padding=(pool.padding, 0, pool.padding, 0),
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
        c_weight = torch.tensor(last_convo.conv.weight.data)
        c_bias = 0. if last_convo.conv.bias is None else torch.tensor(
            last_convo.conv.bias.data)

        # assert last_convo.conv.weight.shape == new_weight.shape
        last_convo.conv.weight.data = c_weight * factor[:, None, None, None]
        last_convo.conv.bias = nn.Parameter((beta + (c_bias - mu) * factor))

    def convert_yolo(self, yolo):
        spk2rates = Spk2Rates(input_shape=self.previous_layer_shape)
        self.add("output_conversion_for_yolo", spk2rates)

        new_yolo = deepcopy(yolo)
        new_yolo.img_dim = 416
        self.spk_mod.add_module(f"yolo_{self.index}", new_yolo)

    def convert_relu(self, relu):
        self.previous_convo().negative_spikes = False

    def convert_zeropad2d(self, padlayer):
        layer = sil.ZeroPad2dLayer(
            image_shape=self.previous_layer_shape,
            padding=padlayer.padding
        )
        self.add("zeropad2d", layer)

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
            elif isinstance(module, nn.ZeroPad2d):
                self.convert_zeropad2d(module)
            elif type(module).__name__ == "YOLOLayer":
                self.convert_yolo(module)
                break
            else:
                warn(f"Layer '{type(module).__name__}' is not supported. Skipping!")

        if self.leftover_rescaling:
            warn("Caution: the rescaling due to the last average pooling could not be applied!")

        return ConvertedNet(self.spk_mod)
