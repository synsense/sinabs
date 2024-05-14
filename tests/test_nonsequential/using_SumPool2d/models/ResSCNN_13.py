import torch.nn as nn
import sinabs.layers as sl
from sinabs.exodus.layers import IAFSqueeze

class SCNN(nn.Module):
    def __init__(self, nb_classes, batch_size, surrogate_fn, min_v_mem=-0.313, spk_thr=2.0) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 16, 3, 1, bias=False)
        self.iaf1 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool1 = sl.SumPool2d(2,2)

        self.conv2 = nn.Conv2d(16, 16, 3, 1, bias=False)
        self.iaf2 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool2 = sl.SumPool2d(2,2)

        self.conv3 = nn.Conv2d(16, 16, 3, 1, bias=False)
        self.iaf3 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool3 = sl.SumPool2d(2,2)

        self.conv4 = nn.Conv2d(16, 16, 3, 1, bias=False)
        self.iaf4 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool4 = sl.SumPool2d(2,2)

        self.conv5 = nn.Conv2d(16, 16, 3, 1, bias=False)
        self.iaf5 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool5 = sl.SumPool2d(2,2)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(64, 200, bias=False)
        self.iaf1_fc = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.fc2 = nn.Linear(200, 200, bias=False)
        self.iaf2_fc = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.fc3 = nn.Linear(200, 200, bias=False)
        self.iaf3_fc = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.fc4 = nn.Linear(200, nb_classes, bias=False)
        self.iaf4_fc = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

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
        rescale_fn(self.conv2, [(2, 2)], lambda_)
        rescale_fn(self.conv3, [(2, 2)], lambda_)
        rescale_fn(self.conv4, [(2, 2)], lambda_)
        rescale_fn(self.conv5, [(2, 2)], lambda_)

    def forward(self, x):
        # conv 1
        con1_out = self.conv1(x)
        iaf1_out = self.iaf1(con1_out)
        pool1_out = self.pool1(iaf1_out)
        # conv 2
        conv2_out = self.conv2(pool1_out)
        iaf2_out = self.iaf2(conv2_out)
        pool2_out = self.pool2(iaf2_out)
        # conv 3
        conv3_out = self.conv3(pool2_out)
        iaf3_out = self.iaf3(conv3_out)
        pool3_out = self.pool3(iaf3_out)
        # conv 4
        conv4_out = self.conv4(pool3_out)
        iaf4_out = self.iaf4(conv4_out)
        pool4_out = self.pool4(iaf4_out)
        # conv 5
        conv5_out = self.conv5(pool4_out)
        iaf5_out = self.iaf5(conv5_out)
        pool5_out = self.pool5(iaf5_out)
        # fc 1
        flat_out = self.flat(pool5_out)
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

        return iaf4_fc_out