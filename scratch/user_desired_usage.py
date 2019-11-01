# This is a scratch file to jot down the desired usage of Speck


####### From a user perspective ##########
import torch


class UserModel(torch.nn.Module):
    """
    Some user define module
    """

    pass


my_model = UserModel()

## Train my model
my_data = []
# Training loop
n_epochs = 10
batches = []
for epoch in range(n_epochs):
    for batch in batches:
        out = my_model(batch)
        # Update parameters
        pass

## Transfer model to SpikingModel
from sinabs.from_torch import from_model

spiking_model = from_model(my_model)

## Test model with spiking data
for spikes in my_data:
    out = spiking_model(spikes)
    acc = None

# Verify that the accuracy of the spiking model is acceptable

# Then port the model to Speck
spiking_model.to("cpu")
spiking_model.to("cuda:0")
spiking_model.to("speck:0")

# Read data from chip
out = spiking_model(my_data)  # Not using the on chip DVS
out = spiking_model(t_read=5)  # Read data for 5 secs
