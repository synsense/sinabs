import torch.nn as nn
import sinabs.layers as sl
from sinabs.exodus.layers import IAFSqueeze

class SCNN(nn.Module):
    def __init__(self, input_size, nb_classes, batch_size, surrogate_fn, min_v_mem=-0.313, spk_thr=2.0) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 8, 2, 1, bias=False)
        self.iaf1 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool1 = sl.SumPool2d(2,2)
        self.pool1a = sl.SumPool2d(4,4)

        self.conv2 = nn.Conv2d(8, 8, 2, 1, bias=False)
        self.iaf2 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool2 = sl.SumPool2d(2,2)

        self.conv3 = nn.Conv2d(8, 8, 2, 1, bias=False)
        self.iaf3 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool3 = sl.SumPool2d(2,2)
        self.pool3a = sl.SumPool2d(4,4)

        self.conv4 = nn.Conv2d(8, 8, 2, 1, bias=False)
        self.iaf4 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool4 = sl.SumPool2d(2,2)

        self.flat = nn.Flatten()

        flat_s = SCNN.get_flatten_size(input_size)

        self.fc1 = nn.Linear(flat_s, 100, bias=False)
        self.iaf1_fc = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.fc2 = nn.Linear(100, 100, bias=False)
        self.iaf2_fc = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.fc3 = nn.Linear(100, 100, bias=False)
        self.iaf3_fc = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.fc4 = nn.Linear(100, 100, bias=False)
        self.iaf4_fc = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.fc5 = nn.Linear(100, nb_classes, bias=False)
        self.iaf5_fc = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        # skip

        self.merge1 = sl.Merge()
        self.merge2 = sl.Merge()

    @staticmethod
    def get_flatten_size(input_size):
        conv1_dims = SCNN.conv2d_output_size(input_size, 8, (2, 2))
        pool1_dims = SCNN.pool_output_size(conv1_dims, 8, (2, 2))

        conv2_dims = SCNN.conv2d_output_size(pool1_dims, 8, (2, 2))
        pool2_dims = SCNN.pool_output_size(conv2_dims, 8, (2, 2))

        conv3_dims = SCNN.conv2d_output_size(pool2_dims, 8, (2, 2))
        pool3_dims = SCNN.pool_output_size(conv3_dims, 8, (2, 2))

        conv4_dims = SCNN.conv2d_output_size(pool3_dims, 8, (2, 2))
        pool4_dims = SCNN.pool_output_size(conv4_dims, 8, (2, 2))

        return pool4_dims[0]*pool4_dims[1]*pool4_dims[2]

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

    def rescale_conv_weights(self, rescale_fn, lambda_):
        rescale_fn(self.conv2, [(2,2)], lambda_)
        rescale_fn(self.conv3, [(4,4), (2,2)], lambda_)
        rescale_fn(self.conv4, [(2,2)], lambda_)

    def forward(self, x):
        # conv 1
        con1_out = self.conv1(x)
        iaf1_out = self.iaf1(con1_out)
        pool1_out = self.pool1(iaf1_out)
        pool1a_out = self.pool1a(iaf1_out)
        # conv 2
        conv2_out = self.conv2(pool1_out)
        iaf2_out = self.iaf2(conv2_out)
        pool2_out = self.pool2(iaf2_out)

        merge1_out = self.merge1(pool1a_out, pool2_out)
        # conv 3
        conv3_out = self.conv3(merge1_out)
        iaf3_out = self.iaf3(conv3_out)
        pool3_out = self.pool3(iaf3_out)
        pool3a_out = self.pool3a(iaf3_out)
        # conv 4
        conv4_out = self.conv4(pool3_out)
        iaf4_out = self.iaf4(conv4_out)
        pool4_out = self.pool4(iaf4_out)

        merge2_out = self.merge2(pool3a_out, pool4_out)

        flat_out = self.flat(merge2_out)
        # fc 1
        fc1_out = self.fc1(flat_out)
        iaf1_fc_out = self.iaf1_fc(fc1_out)
        # fc 2
        fc2_out = self.fc2(iaf1_fc_out)
        iaf2_fc_out = self.iaf2_fc(fc2_out)
        # fc 3
        fc3_out = self.fc3(iaf2_fc_out)
        iaf3_fc_out = self.iaf3_fc(fc3_out)
        # fc 4
        fc4_out = self.fc4(iaf3_fc_out)
        iaf4_fc_out = self.iaf4_fc(fc4_out)
        # fc 5
        fc5_out = self.fc5(iaf4_fc_out)
        iaf5_fc_out = self.iaf5_fc(fc5_out)

        return iaf5_fc_out