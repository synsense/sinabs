from tqdm.notebook import tqdm
import torch

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