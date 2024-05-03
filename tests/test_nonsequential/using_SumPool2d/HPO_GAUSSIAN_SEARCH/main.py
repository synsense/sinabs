from GS_utils import *
import torch, tonic, sys, random
import numpy as np
from torch.utils.data import DataLoader
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm

sys.path.append('../../utils')
sys.path.append('../models')

from network import SCNN_GS
from train_test_fn import training_loop_no_tqdm, load_dataset, split_train_validation
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

max_iter = 100
nb_samples = 5
batch_size = 8
num_workers = 8
validation_ratio = 0.2
n_time_steps = 50
epochs = 40
validation_rand_seed = 1
output_csv = 'gaussian_search_history.csv'
params_set_history = {}
    
loss_fn = CrossEntropyLoss()

hyperparams = {
    'learning_rate': {'value': 0.001, 'min': 0.00008, 'max': 0.08, 'precision': 6, 'sigma': 0.0001},
    'spike_threshold': {'value': 2.75, 'min': 0.5, 'max': 5.0, 'precision': 2, 'sigma': 1.0},
    'mem_v_min': {'value': -2.5, 'min': -5.0, 'max': 0.0, 'precision': 2, 'sigma': 1.0},
    'grad_scale': {'value': 1.55, 'min': 0.1, 'max': 3.0, 'precision': 2, 'sigma': 0.5},
    'grad_width': {'value': 1.55, 'min': 0.1, 'max': 3.0, 'precision': 2, 'sigma': 0.5},
    'w_rescale_lambda': {'value': 0.5, 'min': 0.1, 'max': 1.0, 'precision': 3, 'sigma': 0.1667},
}

prob_model = define_probabilistic_model(hyperparams)

with open('fixed_parameters.txt', 'w') as file:
    file.write(f'max_iter: {max_iter}\n')
    file.write(f'nb_samples: {nb_samples}\n')
    file.write(f'batch_size: {batch_size}\n')
    file.write(f'num_workers: {num_workers}\n')
    file.write(f'validation_ratio: {validation_ratio}\n')
    file.write(f'n_time_steps: {n_time_steps}\n')
    file.write(f'epochs: {epochs}\n')
    file.write(f'validation_rand_seed: {validation_rand_seed}\n')

### Data Loading #####################################################

snn_train_dataset, snn_test_dataset, sensor_size = load_dataset('DVSGESTURE', n_time_steps)
train_dataset, validation_dataset = split_train_validation(validation_ratio, snn_train_dataset, validation_rand_seed)

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

### Baseline Accuracy ################################################

# instantiate model.
csnn = SCNN_GS(
    batch_size = batch_size,
    surrogate_fn = PeriodicExponential(grad_scale = hyperparams['grad_scale']['value'], grad_width = hyperparams['grad_width']['value']),
    min_v_mem = hyperparams['mem_v_min']['value'],
    spk_thr = hyperparams['spike_threshold']['value'],
    rescale_fn = rescale_method_1,
    rescale_lambda = hyperparams['w_rescale_lambda']['value']
    ).to(device)

# instantiate optimizer.
optimizer = Adam(csnn.parameters(), lr = hyperparams['learning_rate']['value'], betas = (0.9, 0.999), eps = 1e-8)

# train/test model.
best_acc = training_loop_no_tqdm(
    device, 
    n_time_steps,
    batch_size,
    sensor_size,
    snn_train_dataloader, 
    csnn, 
    loss_fn, 
    optimizer, 
    epochs, 
    snn_validation_dataloader)

# initialize parameters history.
best_param_set = {
    'learning_rate': [hyperparams['learning_rate']['value']],
    'spike_threshold': [hyperparams['spike_threshold']['value']],
    'mem_v_min': [hyperparams['mem_v_min']['value']],
    'grad_scale': [hyperparams['grad_scale']['value']],
    'grad_width': [hyperparams['grad_width']['value']],
    'w_rescale_lambda': [hyperparams['w_rescale_lambda']['value']],
    'accuracy': [best_acc]
}

df = pd.DataFrame(best_param_set)
df.to_csv(output_csv, index=True)

print(f'> initial accuracy: {best_acc}\n')

### HPO Loop ##########################################################

train_p_bar = tqdm(range(1, max_iter+1))
counter = 1

for iter in train_p_bar:

    # sample values to be tested.
    sampled_values = sample_values_to_eval(iter, prob_model, hyperparams, nb_samples)

    # test each sampled set.
    acc = []
    for i in range(nb_samples):
        sampled_set = get_sampled_set(sampled_values, i)

        # instantiate model.
        csnn = SCNN_GS(
            batch_size = batch_size,
            surrogate_fn = PeriodicExponential(grad_scale = sampled_set['grad_scale'], grad_width = sampled_set['grad_width']),
            min_v_mem = sampled_set['mem_v_min'],
            spk_thr = sampled_set['spike_threshold'],
            rescale_fn = rescale_method_1,
            rescale_lambda = sampled_set['w_rescale_lambda']
            ).to(device)
        
        # instantiate optimizer.
        optimizer = Adam(csnn.parameters(), lr = sampled_set['learning_rate'], betas = (0.9, 0.999), eps = 1e-8)

        # train/test model.
        ith_acc = training_loop_no_tqdm(
            device, 
            n_time_steps,
            batch_size,
            sensor_size,
            snn_train_dataloader, 
            csnn, 
            loss_fn, 
            optimizer, 
            epochs, 
            snn_validation_dataloader)
        
        acc.append(ith_acc)

        # update progress bar
        train_p_bar.set_description(f'model {counter}/{max_iter*nb_samples} - best acc.: {np.round(best_acc, 2)}')
        counter += 1

    # get best parameters set.
    highest_acc_index = acc.index(np.max(acc))
    best_param_set = get_sampled_set(sampled_values, highest_acc_index)

    # update model.
    if acc[highest_acc_index] > best_acc:
        best_acc = acc[highest_acc_index]
        
        update_probabilistic_model(max_iter, iter, prob_model, best_param_set)

        # save to history.
        best_param_set['accuracy'] = best_acc

    else:

        best_param_set = {}
        for key, val in prob_model.items():
            best_param_set[key] = val['mu']
        best_param_set['accuracy'] = best_acc

    # update history
    df = pd.DataFrame([best_param_set], index=[iter])
    df.to_csv(output_csv, mode='a', header=False)