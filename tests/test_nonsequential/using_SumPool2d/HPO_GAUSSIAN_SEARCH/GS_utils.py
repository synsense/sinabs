import numpy as np

def define_probabilistic_model(hyperparams):
    prob_model = {}

    for key, data in hyperparams.items():
        prob_model[key] = {'mu': data['value'], 'sigma': data['sigma']}

    return prob_model

def update_probabilistic_model(max_iter, cur_iter, prob_model, hyperparams_new):

    sigma_scale = update_sigma_scale(max_iter, cur_iter)

    for key, value in hyperparams_new.items():
        prob_model[key]['mu'] = value
        prob_model[key]['sigma'] = prob_model[key]['sigma']*sigma_scale

def update_sigma_scale(max_iter, cur_iter):
    _ = np.round(1-(cur_iter/max_iter), 2)

    if _ > 0:
        return _
    else:
        return 0.01
    
def sample_values_to_eval(iteration, prob_model, hyperparams, nb_samples: int = 5):
    new_values = {}
    np.random.seed(iteration)

    for hp, data in prob_model.items():
        sampled_values = np.round(np.random.normal(data['mu'], data['sigma'], nb_samples), hyperparams[hp]['precision'])

        fixed_sampled_values = []

        for val in sampled_values:
            if val < hyperparams[hp]['min']:
                fixed_sampled_values.append(hyperparams[hp]['min'])
            elif val > hyperparams[hp]['max']:
                fixed_sampled_values.append(hyperparams[hp]['max'])
            else:
                fixed_sampled_values.append(val)

        new_values[hp] = fixed_sampled_values

    return new_values

def get_sampled_set(sampled_values, i):
    sampled_set = {}

    for key, val in sampled_values.items():
        sampled_set[key] = val[i]

    return sampled_set