import torch, random, sys
import torch.nn as nn
from tqdm.notebook import tqdm

from tonic.transforms import ToFrame
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import numpy as np

from nonseq_model import SNN

torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

batch_size = 3
num_workers = 1
epochs = 30
lr = 1e-3
n_time_steps = 50

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('device: ', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')

def train(batch_size, feature_map_size, dataloader_train, model, loss_fn, optimizer, epochs, test_func, dataloader_test, phase):
    epochs_y = []
    epochs_x = []
    epochs_acc = []
    model.train()

    for e in range(epochs):
        losses = []
        batches = []
        batch_count = 0
        train_p_bar = tqdm(dataloader_train)

        for X, y in train_p_bar:
            # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
            X = X.reshape(-1, feature_map_size[2], feature_map_size[0], feature_map_size[1]).to(dtype=torch.float, device=device)
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

            train_p_bar.set_description(f"{phase} - Epoch {e} - BPTT Training Loss: {round(loss.item(), 4)}")

            batch_count += 1
            losses.append(loss.item())
            batches.append(batch_count)

        epochs_y.append(losses)
        epochs_x.append(batches)

        acc = test_func(batch_size, feature_map_size, dataloader_test, model)
        print(f'{phase} - Epoch {e} accuracy: {acc}')
        epochs_acc.append(acc)

    return epochs_x, epochs_y, epochs_acc

def test(batch_size, feature_map_size, dataloader, model):
    correct_predictions = []
    with torch.no_grad():
        test_p_bar = tqdm(dataloader)
        for X, y in test_p_bar:
            # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
            X = X.reshape(-1, feature_map_size[2], feature_map_size[0], feature_map_size[1]).to(dtype=torch.float, device=device)
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

            test_p_bar.set_description(f"Testing Model...")
    
    correct_predictions = torch.cat(correct_predictions)
    return correct_predictions.sum().item()/(len(correct_predictions))*100

from tonic.datasets.dvsgesture import DVSGesture

root_dir = "../DVSGESTURE"
_ = DVSGesture(save_to=root_dir, train=True)
_ = DVSGesture(save_to=root_dir, train=False)

to_raster = ToFrame(sensor_size=DVSGesture.sensor_size, n_time_bins=n_time_steps)

snn_train_dataset = DVSGesture(save_to=root_dir, train=True, transform=to_raster)
snn_test_dataset = DVSGesture(save_to=root_dir, train=False, transform=to_raster)

sample_data, label = snn_train_dataset[0]
print(f"The transformed array is in shape [Time-Step, Channel, Height, Width] --> {sample_data.shape}")

snn_train_dataloader = DataLoader(snn_train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)
snn_test_dataloader = DataLoader(snn_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

snn = SNN(11, 810, batch_size).to(device)
snn.init_weights()

snn.load_conv_params(int(sys.argv[1]))

optimizer = Adam(snn.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
loss_fn = CrossEntropyLoss()

epochs_x, epochs_y, epochs_acc = train(
    batch_size,
    DVSGesture.sensor_size, 
    snn_train_dataloader, 
    snn, 
    loss_fn, 
    optimizer, 
    epochs, 
    test, 
    snn_test_dataloader,
    'post-training'
    )

with open(f'nonseq_TA1_w_load_{sys.argv[1]}_training_metrics.npy', 'wb') as f:
    np.save(f, np.array(epochs_x))
    np.save(f, np.array(epochs_y))
    np.save(f, np.array(epochs_acc))