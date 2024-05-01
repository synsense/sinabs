import torch.nn as nn
import sinabs.layers as sl
from sinabs.exodus.layers import IAFSqueeze

class ResCSNN3(nn.Module):
    def __init__(self, input_size, nb_classes, batch_size, surrogate_fn, min_v_mem=-1.0, spk_thr=1.0) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 10, 2, 1, bias=False)
        self.iaf1 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool1 = sl.SumPool2d(2,2)
        self.pool1a = sl.SumPool2d(6,6)

        self.conv2 = nn.Conv2d(10, 10, 2, 1, bias=False)
        self.iaf2 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool2 = sl.SumPool2d(3,3)

        self.conv3 = nn.Conv2d(10, 10, 3, 1, bias=False)
        self.iaf3 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool3 = sl.SumPool2d(2,2)

        self.flat = nn.Flatten()

        flat_s = ResCSNN3.get_flatten_size(input_size)

        self.fc1 = nn.Linear(flat_s, 100, bias=False)
        self.iaf4 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.fc2 = nn.Linear(100, 100, bias=False)
        self.iaf5 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.fc3 = nn.Linear(100, 100, bias=False)
        self.iaf6 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.fc4 = nn.Linear(100, nb_classes, bias=False)
        self.iaf7 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.merge_fc = sl.Merge()
        self.merge_conv = sl.Merge()

    @staticmethod
    def get_flatten_size(input_size):
        conv1_dims = ResCSNN3.conv2d_output_size(input_size, 10, (2, 2))
        pool1_dims = ResCSNN3.pool_output_size(conv1_dims, 10, (2, 2))

        conv2_dims = ResCSNN3.conv2d_output_size(pool1_dims, 10, (2, 2))
        pool2_dims = ResCSNN3.pool_output_size(conv2_dims, 10, (3, 3))

        conv3_dims = ResCSNN3.conv2d_output_size(pool2_dims, 10, (3, 3))
        pool3_dims = ResCSNN3.pool_output_size(conv3_dims, 10, (2, 2))

        return pool3_dims[0]*pool3_dims[1]*pool3_dims[2]

    @staticmethod
    def conv2d_output_size(input_size, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        input_height, input_width, input_channels = input_size
        kernel_height, kernel_width = kernel_size

        output_height = ((input_height + 2 * padding - dilation * (kernel_height - 1) - 1) // stride) + 1
        output_width = ((input_width + 2 * padding - dilation * (kernel_width - 1) - 1) // stride) + 1

        return (output_height, output_width, out_channels)

    @staticmethod
    def pool_output_size(input_size, out_channels, kernel_size, stride=None, padding=0):
        input_height, input_width, input_channels = input_size
        kernel_height, kernel_width = kernel_size

        if stride is None:
            stride = kernel_height

        output_height = ((input_height + 2 * padding - kernel_height) // stride) + 1
        output_width = ((input_width + 2 * padding - kernel_width) // stride) + 1

        return (output_height, output_width, out_channels)

    def detach_neuron_states(self):
        for name, layer in self.named_modules():
            if name != '':
                if isinstance(layer, sl.StatefulLayer):
                    for name, buffer in layer.named_buffers():
                        buffer.detach_()

    def init_weights(self):
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data)

    def forward(self, x):
        
        con1_out = self.conv1(x)
        iaf1_out = self.iaf1(con1_out)
        pool1_out = self.pool1(iaf1_out)
        pool1a_out = self.pool1a(iaf1_out)

        conv2_out = self.conv2(pool1_out)
        iaf2_out = self.iaf2(conv2_out)
        pool2_out = self.pool2(iaf2_out)

        merged_conv_out = self.merge_conv(pool1a_out, pool2_out)

        conv3_out = self.conv3(merged_conv_out)
        iaf3_out = self.iaf3(conv3_out)
        pool3_out = self.pool3(iaf3_out)

        flat_out = self.flat(pool3_out)
        
        fc1_out = self.fc1(flat_out)
        iaf4_out = self.iaf4(fc1_out)

        fc2_out = self.fc2(iaf4_out)
        iaf5_out = self.iaf5(fc2_out)

        fc3_out = self.fc3(iaf5_out)
        iaf6_out = self.iaf6(fc3_out)

        merge_fc_out = self.merge_fc(iaf4_out, iaf6_out)

        fc4_out = self.fc4(merge_fc_out)
        iaf7_out = self.iaf7(fc4_out)

        return iaf7_out