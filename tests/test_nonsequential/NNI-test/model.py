import torch
import torch.nn as nn
import sinabs.layers as sl
import nni

from tonic.datasets.nmnist import NMNIST
from tonic.transforms import ToFrame
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

params = {
    'lr': 0.001,
}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)
print(params)

###### Loading Data ######

batch_size = 32
num_workers = 4
epochs = 1

root_dir = "./NMNIST"
_ = NMNIST(save_to=root_dir, train=True)
_ = NMNIST(save_to=root_dir, train=False)

n_time_steps = 50
to_raster = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=n_time_steps)

snn_train_dataset = NMNIST(save_to=root_dir, train=True, transform=to_raster)
snn_test_dataset = NMNIST(save_to=root_dir, train=False, transform=to_raster)

snn_train_dataloader = DataLoader(snn_train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)
snn_test_dataloader = DataLoader(snn_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

###### Defining the Model ######

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('device: ', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')

class SNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 10, 2, 1, bias=False) 
        self.iaf1 = sl.IAFSqueeze(batch_size=1)            
        self.pool1 = sl.SumPool2d(3,3)                  
        self.pool1a = sl.SumPool2d(4,4)                 

        self.conv2 = nn.Conv2d(10, 10, 4, 1, bias=False)
        self.iaf2 = sl.IAFSqueeze(batch_size=1)            

        self.conv3 = nn.Conv2d(10, 1, 2, 1, bias=False) 
        self.iaf3 = sl.IAFSqueeze(batch_size=1)            

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(49, 100, bias=False)       
        self.iaf4 = sl.IAFSqueeze(batch_size=1)            
        
        self.fc2 = nn.Linear(100, 10, bias=False)       
        self.iaf5 = sl.IAFSqueeze(batch_size=1)            


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

        conv3_out = self.conv3(iaf2_out)
        iaf3_out = self.iaf3(conv3_out)

        flat_out = self.flat(iaf3_out)
        
        fc1_out = self.fc1(flat_out)
        iaf4_out = self.iaf4(fc1_out)
        fc2_out = self.fc2(iaf4_out)
        iaf5_out = self.iaf5(fc2_out)

        return iaf5_out
    
snn = SNN().to(device)

snn.init_weights()

optimizer = SGD(snn.parameters(), lr=params['lr'])
loss_fn = CrossEntropyLoss()

###### Defining Train/Test ######

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
        X = X.reshape(-1, 2, 34, 34).to(dtype=torch.float, device=device)
        y = y.to(dtype=torch.long, device=device)

        # forward
        pred = model(X)

        # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
        pred = pred.reshape(batch_size, n_time_steps, -1)

        # accumulate all time-steps output for final prediction
        pred = pred.sum(dim = 1)
        loss = loss_fn(pred, y)

        # gradient update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # detach the neuron states and activations from current computation graph(necessary)
        model.detach_neuron_states()

        break

def test(dataloader, model):
    correct_predictions = []
    with torch.no_grad():
        for X, y in dataloader:
            # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
            X = X.reshape(-1, 2, 34, 34).to(dtype=torch.float, device=device)
            y = y.to(dtype=torch.long, device=device)

            # forward
            output = model(X)

            # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
            output = output.reshape(batch_size, n_time_steps, -1)

            # accumulate all time-steps output for final prediction
            output = output.sum(dim=1)

            # calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)

            # compute the total correct predictions
            correct_predictions.append(pred.eq(y.view_as(pred)))

            break

    return correct_predictions.sum().item()/(len(correct_predictions))*100

###### Training loop (HPO) ######

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(snn_train_dataloader, snn, loss_fn, optimizer)
    accuracy = test(snn_test_dataloader, snn)
    nni.report_intermediate_result(accuracy)
nni.report_final_result(accuracy)