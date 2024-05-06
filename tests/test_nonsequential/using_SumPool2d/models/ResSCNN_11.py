import torch.nn as nn
import sinabs.layers as sl
from sinabs.exodus.layers import IAFSqueeze

class SCNN(nn.Module):
    def __init__(self, input_size, nb_classes, batch_size, surrogate_fn, min_v_mem=-0.313, spk_thr=2.0) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 8, 2, 1, bias=False)
        self.iaf1 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.conv2 = nn.Conv2d(8, 8, 3, 1, 1, bias=False)
        self.iaf2 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.conv3 = nn.Conv2d(8, 8, 3, 1, 1, bias=False)
        self.iaf3 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool3a = sl.SumPool2d(4,4)

        self.conv4 = nn.Conv2d(8, 8, 2, 1, bias=False)
        self.iaf4 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool4 = sl.SumPool2d(2,2)

        self.conv5 = nn.Conv2d(8, 8, 2, 1, bias=False)
        self.iaf5 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool5 = sl.SumPool2d(2,2)
        self.pool5a = sl.SumPool2d(4,4)

        self.conv6 = nn.Conv2d(8, 8, 3, 1, 1, bias=False)
        self.iaf6 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        self.conv7 = nn.Conv2d(8, 8, 2, 1, bias=False)
        self.iaf7 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool7 = sl.SumPool2d(2,2)

        self.conv8 = nn.Conv2d(8, 8, 2, 1, bias=False)
        self.iaf8 = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)
        self.pool8 = sl.SumPool2d(2,2)

        self.flat = nn.Flatten()

        self.fc_out = nn.Linear(392, nb_classes, bias=False)
        self.iaf_fc_out = IAFSqueeze(batch_size=batch_size, min_v_mem=min_v_mem, surrogate_grad_fn=surrogate_fn, spike_threshold=spk_thr)

        # skip

        self.merge1 = sl.Merge()
        self.merge2 = sl.Merge()
        self.merge3 = sl.Merge()
        self.merge4 = sl.Merge()

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
        rescale_fn(self.conv5, [(2,2)], lambda_)
        rescale_fn(self.conv6, [(4,4), (2,2)], lambda_)
        rescale_fn(self.conv8, [(4,4), (2,2)], lambda_)

    def forward(self, x):
        #  -- conv block 1 ---
        con1_out = self.conv1(x)
        iaf1_out = self.iaf1(con1_out)        # to CONV 4
        #  -- conv block 2 ---
        conv2_out = self.conv2(iaf1_out)
        iaf2_out = self.iaf2(conv2_out)       
        #  -- conv block 3 ---
        conv3_out = self.conv3(iaf2_out)
        iaf3_out = self.iaf3(conv3_out)
        pool3a_out = self.pool3a(iaf3_out)    # to CONV 6
        #  -- conv block 4 ---
        #print(iaf1_out.shape, iaf3_out.shape)
        merge1_out = self.merge1(iaf1_out, iaf3_out)
        conv4_out = self.conv4(merge1_out)
        iaf4_out = self.iaf4(conv4_out)
        pool4_out = self.pool4(iaf4_out)
        #  -- conv block 5 ---
        conv5_out = self.conv5(pool4_out)
        iaf5_out = self.iaf5(conv5_out)
        pool5_out = self.pool5(iaf5_out)
        pool5a_out = self.pool5a(iaf5_out)    # to CONV 8
        #  -- conv block 6 ---
        #print(pool3a_out.shape, pool5_out.shape)
        merge3_out = self.merge3(pool3a_out, pool5_out)
        conv6_out = self.conv6(merge3_out)
        iaf6_out = self.iaf6(conv6_out)
        #  -- conv block 7 ---
        conv7_out = self.conv7(iaf6_out)
        iaf7_out = self.iaf7(conv7_out)
        pool7_out = self.pool7(iaf7_out)
        #  -- conv block 8 ---
        #print(pool5a_out.shape, pool7_out.shape)
        merge4_out = self.merge4(pool5a_out, pool7_out)
        conv8_out = self.conv8(merge4_out)
        iaf8_out = self.iaf8(conv8_out)
        pool8_out = self.pool8(iaf8_out)
        # -- output --
        flat = self.flat(pool8_out)
        #print(flat.shape)
        fc_out = self.fc_out(flat)
        iaf_fc_out = self.iaf_fc_out(fc_out)
    

        return iaf_fc_out