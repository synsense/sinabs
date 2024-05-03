from tqdm.notebook import tqdm
import torch
from tonic.datasets.dvsgesture import DVSGesture
from tonic.datasets.nmnist import NMNIST
from tonic.transforms import ToFrame
import numpy as np
from torch.utils.data import Subset

def training_loop(device, nb_time_steps, batch_size, feature_map_size, dataloader_train, model, loss_fn, optimizer, epochs, dataloader_test):
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
            pred = pred.reshape(batch_size, nb_time_steps, -1)

            # accumulate all time-steps output for final prediction
            pred = pred.sum(dim = 1)
            loss = loss_fn(pred, y)

            # gradient update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # detach the neuron states and activations from current computation graph(necessary)
            model.detach_neuron_states()

            train_p_bar.set_description(f"Epoch {e} - BPTT Training Loss: {round(loss.item(), 4)}")

            batch_count += 1
            losses.append(loss.item())
            batches.append(batch_count)

        epochs_y.append(losses)
        epochs_x.append(batches)

        acc = test(device, nb_time_steps, batch_size, feature_map_size, dataloader_test, model)
        print(f'Epoch {e} accuracy: {acc}')
        epochs_acc.append(acc)

    return epochs_x, epochs_y, epochs_acc

def training_loop_no_tqdm(device, nb_time_steps, batch_size, feature_map_size, dataloader_train, model, loss_fn, optimizer, epochs, dataloader_test):
    model.train()

    for e in range(epochs):
        for X, y in dataloader_train:
            # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
            X = X.reshape(-1, feature_map_size[2], feature_map_size[0], feature_map_size[1]).to(dtype=torch.float, device=device)
            y = y.to(dtype=torch.long, device=device)

            # forward
            pred = model(X)

            # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
            pred = pred.reshape(batch_size, nb_time_steps, -1)

            # accumulate all time-steps output for final prediction
            pred = pred.sum(dim = 1)
            loss = loss_fn(pred, y)

            # gradient update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # detach the neuron states and activations from current computation graph(necessary)
            model.detach_neuron_states()

        acc = test_no_tqdm(device, nb_time_steps, batch_size, feature_map_size, dataloader_test, model)

    return acc

def test_no_tqdm(device, nb_time_steps, batch_size, feature_map_size, dataloader_test, model):
    correct_predictions = []

    with torch.no_grad():
        for X, y in dataloader_test:
            # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
            X = X.reshape(-1, feature_map_size[2], feature_map_size[0], feature_map_size[1]).to(dtype=torch.float, device=device)
            y = y.to(dtype=torch.long, device=device)

            # forward
            output = model(X)

            # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
            output = output.reshape(batch_size, nb_time_steps, -1)

            # accumulate all time-steps output for final prediction
            output = output.sum(dim=1)

            # calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)

            # compute the total correct predictions
            correct_predictions.append(pred.eq(y.view_as(pred)))
    
    correct_predictions = torch.cat(correct_predictions)
    return correct_predictions.sum().item()/(len(correct_predictions))*100

def test(device, nb_time_steps, batch_size, feature_map_size, dataloader_test, model):
    correct_predictions = []
    with torch.no_grad():
        test_p_bar = tqdm(dataloader_test)
        for X, y in test_p_bar:
            # reshape the input from [Batch, Time, Channel, Height, Width] into [Batch*Time, Channel, Height, Width]
            X = X.reshape(-1, feature_map_size[2], feature_map_size[0], feature_map_size[1]).to(dtype=torch.float, device=device)
            y = y.to(dtype=torch.long, device=device)

            # forward
            output = model(X)

            # reshape the output from [Batch*Time,num_classes] into [Batch, Time, num_classes]
            output = output.reshape(batch_size, nb_time_steps, -1)

            # accumulate all time-steps output for final prediction
            output = output.sum(dim=1)

            # calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)

            # compute the total correct predictions
            correct_predictions.append(pred.eq(y.view_as(pred)))

            test_p_bar.set_description(f"Testing Model...")
    
    correct_predictions = torch.cat(correct_predictions)
    return correct_predictions.sum().item()/(len(correct_predictions))*100

def load_dataset(dataset, n_time_steps):
    if dataset == 'DVSGESTURE':
        root_dir = "../../DVSGESTURE"
        _ = DVSGesture(save_to=root_dir, train=True)
        _ = DVSGesture(save_to=root_dir, train=False)

        to_raster = ToFrame(sensor_size=DVSGesture.sensor_size, n_time_bins=n_time_steps)

        snn_train_dataset = DVSGesture(save_to=root_dir, train=True, transform=to_raster)
        snn_test_dataset = DVSGesture(save_to=root_dir, train=False, transform=to_raster)

        return snn_train_dataset, snn_test_dataset, DVSGesture.sensor_size
    
    elif dataset == 'NMNIST':
        root_dir = "../../NMNIST"
        _ = NMNIST(save_to=root_dir, train=True)
        _ = NMNIST(save_to=root_dir, train=False)

        to_raster = ToFrame(sensor_size=NMNIST.sensor_size, n_time_bins=n_time_steps)

        snn_train_dataset = NMNIST(save_to=root_dir, train=True, transform=to_raster)
        snn_test_dataset = NMNIST(save_to=root_dir, train=False, transform=to_raster)

        return snn_train_dataset, snn_test_dataset, NMNIST.sensor_size
    
    else:

        raise ValueError('no valid dataset')
    
def split_train_validation(validation_ratio, snn_train_dataset, rand_seed):
    num_samples = len(snn_train_dataset)
    num_validation_samples = int(validation_ratio * num_samples)

    np.random.seed(rand_seed)

    validation_indices = np.random.choice(np.arange(num_samples), size=num_validation_samples, replace=False)
    training_indices = np.array(list(filter(lambda x: x not in validation_indices, np.arange(num_samples))))

    train_dataset = Subset(snn_train_dataset, training_indices)
    validation_dataset = Subset(snn_train_dataset, validation_indices)

    return train_dataset, validation_dataset

def load_architecture(architecture, input_size, nb_classes, batch_size, surrogate_fn, min_v_mem=-0.313, spk_thr=2.0):
    import sys
    sys.path.append('../models')

    if architecture == 'ResSCNN1':
        from ResSCNN1 import SCNN
        return SCNN(input_size, nb_classes, batch_size, surrogate_fn, min_v_mem, spk_thr)
    elif architecture == 'ResSCNN2':
        from ResSCNN2 import SCNN
        return SCNN(input_size, nb_classes, batch_size, surrogate_fn, min_v_mem, spk_thr)
    elif architecture == 'ResSCNN3':
        from ResSCNN3 import SCNN
        return SCNN(input_size, nb_classes, batch_size, surrogate_fn, min_v_mem, spk_thr)
    elif architecture == 'ResSCNN4':
        from ResSCNN4 import SCNN
        return SCNN(input_size, nb_classes, batch_size, surrogate_fn, min_v_mem, spk_thr)
    elif architecture == 'ResSCNN5':
        from ResSCNN5 import SCNN
        return SCNN(input_size, nb_classes, batch_size, surrogate_fn, min_v_mem, spk_thr)
    elif architecture == 'ResSCNN6':
        from ResSCNN6 import SCNN
        return SCNN(input_size, nb_classes, batch_size, surrogate_fn, min_v_mem, spk_thr)
    elif architecture == 'ResSCNN7':
        from ResSCNN7 import SCNN
        return SCNN(input_size, nb_classes, batch_size, surrogate_fn, min_v_mem, spk_thr)
    elif architecture == 'ResSCNN8':
        from ResSCNN8 import SCNN
        return SCNN(input_size, nb_classes, batch_size, surrogate_fn, min_v_mem, spk_thr)
    else:
        return None