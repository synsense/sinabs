import torch, tonic, sys, random
import numpy as np
from torch.utils.data import DataLoader
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
from tqdm import tqdm

sys.path.append('../../utils')
sys.path.append('../models')

from train_test_fn import training_loop_no_tqdm, load_dataset, split_train_validation_used_seed, load_architecture
from weight_initialization import rescale_method_1

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('device: ', torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')

torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

### Initialization ####################################################

total_architectures = 11

lr = 5e-5
batch_size = 8
num_workers = 4
n_time_steps = 50
epochs = 25
w_rescale_lambda = 0.8

spk_thr = 2.0
v_min = -0.313
grad_scale = 1.534
grad_width = 0.759

validation_ratio = 0.2
prev_used_seed = 1

loss_fn = CrossEntropyLoss()

directory = f'./architectures_results_2'

if not os.path.exists(directory):
    os.makedirs(directory)

with open(f'./architectures_results_2/fixed_parameters.txt', 'w') as file:
    file.write(f'lr: {lr}\n')
    file.write(f'batch_size: {batch_size}\n')
    file.write(f'num_workers: {num_workers}\n')
    file.write(f'n_time_steps: {n_time_steps}\n')
    file.write(f'epochs: {epochs}\n')
    file.write(f'w_rescale_lambda: {w_rescale_lambda}\n')
    file.write(f'spk_thr: {spk_thr}\n')
    file.write(f'v_min: {v_min}\n')
    file.write(f'grad_scale: {grad_scale}\n')
    file.write(f'grad_width: {grad_width}\n')
    file.write(f'validation_ratio: {validation_ratio}\n')
    file.write(f'prev_used_seed: {prev_used_seed}\n')

### Data Loading #####################################################

snn_train_dataset, snn_test_dataset, sensor_size = load_dataset('DVSGESTURE', n_time_steps)
train_dataset, validation_dataset = split_train_validation_used_seed(validation_ratio, snn_train_dataset, prev_used_seed)

disk_cache_train = tonic.DiskCachedDataset(
    dataset=train_dataset,
    cache_path='./cached_train'
)
snn_train_dataloader = DataLoader(disk_cache_train, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)

disk_cache_validation = tonic.DiskCachedDataset(
    dataset=validation_dataset,
    cache_path='./cached_validation'
)
snn_validation_dataloader = DataLoader(disk_cache_validation, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)

disk_cache_test = tonic.DiskCachedDataset(
    dataset=snn_test_dataset,
    cache_path='./cached_test'
)
snn_test_dataloader = DataLoader(disk_cache_test, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

### Training Loop ##########################################################

train_p_bar = tqdm(range(9, total_architectures+1))

for iter in train_p_bar:
    achitecture = f'ResSCNN_{iter}'

    # instantiate model.
    csnn = load_architecture(
        achitecture, 
        sensor_size, 
        11,
        batch_size, 
        PeriodicExponential(grad_scale=grad_scale, grad_width=grad_width), 
        v_min, 
        spk_thr
        ).to(device)

    csnn.init_weights()
    csnn.rescale_conv_weights(rescale_method_1, w_rescale_lambda)
    
    # instantiate optimizer.
    optimizer = Adam(csnn.parameters(), lr = lr, betas = (0.9, 0.999), eps = 1e-8)

    # train/test model.
    epochs_x, epochs_y, epochs_acc = training_loop_no_tqdm(
        device, 
        n_time_steps,
        batch_size,
        sensor_size,
        snn_train_dataloader, 
        csnn, 
        loss_fn, 
        optimizer, 
        epochs, 
        snn_validation_dataloader,
        True)
    
    # export model data.
    with open(f'./architectures_results_2/{achitecture}-training_metrics.npy', 'wb') as f:
        np.save(f, np.array(epochs_x))
        np.save(f, np.array(epochs_y))
        np.save(f, np.array(epochs_acc))

    # update progress bar
    train_p_bar.set_description(f'{iter}/{total_architectures} - model {achitecture} - acc.: {np.round(epochs_acc[-1], 2)}')