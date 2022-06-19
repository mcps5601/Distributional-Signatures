import itertools
import numpy as np
import scipy.stats
import math


def named_grad_param(model, keys):
    '''
        Return a generator that generates learnable named parameters in
        model[key] for key in keys.
    '''
    if len(keys) == 1:
        return filter(lambda p: p[1].requires_grad,
                model[keys[0]].named_parameters())
    else:
        return filter(lambda p: p[1].requires_grad,
                itertools.chain.from_iterable(
                    model[key].named_parameters() for key in keys))


def grad_param(model, keys):
    '''
        Return a generator that generates learnable parameters in
        model[key] for key in keys.
    '''
    if len(keys) == 1:
        return filter(lambda p: p.requires_grad,
                model[keys[0]].parameters())
    else:
        return filter(lambda p: p.requires_grad,
                itertools.chain.from_iterable(
                    model[key].parameters() for key in keys))


def get_norm(model):
    '''
        Compute norm of the gradients
    '''
    total_norm = 0

    for p in model.parameters():
        if p.grad is not None:
            p_norm = p.grad.data.norm()
            total_norm += p_norm.item() ** 2

    total_norm = total_norm ** 0.5

    return total_norm


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h


def mean_confidence_interval_known(data, confidence=0.95):
    return 1.96 * np.std(data) / math.sqrt(len(data))