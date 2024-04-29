import torch
import torch.nn as nn
import sinabs.layers as sl
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential

class SNN(nn.Module):
    def __init__(self, nb_classes, pool2lin_size, batch_size) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 10, 2, 1, bias=False)
        self.iaf1 = sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential())
        self.pool1 = nn.AvgPool2d(2,2)

        self.conv2 = nn.Conv2d(10, 10, 2, 1, bias=False)
        self.iaf2 = sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential())
        self.pool2 = nn.AvgPool2d(3,3)

        self.conv3 = nn.Conv2d(10, 10, 3, 1, bias=False)
        self.iaf3 = sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential())
        self.pool3 = nn.AvgPool2d(2,2)

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(pool2lin_size, 100, bias=False)
        self.iaf4 = sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential())

        self.fc2 = nn.Linear(100, 100, bias=False)
        self.iaf5 = sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential())

        self.fc3 = nn.Linear(100, 100, bias=False)
        self.iaf6 = sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential())

        self.fc4 = nn.Linear(100, nb_classes, bias=False)
        self.iaf7 = sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential())

    def export_conv_params(self):
        torch.save(self.conv1.state_dict(), 'seq_conv1_weights.pth')
        torch.save(self.conv2.state_dict(), 'seq_conv2_weights.pth')
        torch.save(self.conv3.state_dict(), 'seq_conv3_weights.pth')

    def load_conv_params(self):
        self.conv1.load_state_dict(torch.load('seq_conv1_weights.pth'))
        self.conv2.load_state_dict(torch.load('seq_conv2_weights.pth'))
        self.conv3.load_state_dict(torch.load('seq_conv3_weights.pth'))

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

        conv2_out = self.conv2(pool1_out)
        iaf2_out = self.iaf2(conv2_out)
        pool2_out = self.pool2(iaf2_out)

        conv3_out = self.conv3(pool2_out)
        iaf3_out = self.iaf3(conv3_out)
        pool3_out = self.pool3(iaf3_out)

        flat_out = self.flat(pool3_out)
        
        fc1_out = self.fc1(flat_out)
        iaf4_out = self.iaf4(fc1_out)

        fc2_out = self.fc2(iaf4_out)
        iaf5_out = self.iaf5(fc2_out)

        fc3_out = self.fc3(iaf5_out)
        iaf6_out = self.iaf6(fc3_out)

        fc4_out = self.fc4(iaf6_out)
        iaf7_out = self.iaf7(fc4_out)

        return iaf7_out