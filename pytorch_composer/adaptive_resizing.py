from torch import nn, rand
import numpy as np
import math


def dimensions_inference(inpu, oupu):
    input_features = np.prod(inpu)
    output_fixed = 1
    output_variable_factor = 1
    variables_index = []
    variables = []
    for i, dim in enumerate(oupu):
        if dim > 0:
            output_fixed *= dim
        elif dim < 0:
            output_variable_factor *= -dim
            variables_index.append(i)
            variables.append(-dim)
    if variables == [] or variables == [-1]:
        return oupu

    output_free = input_features / output_fixed
    free_ranks = len(variables_index)
    n = (output_free / output_variable_factor)**(1 / free_ranks)
    if not n.is_integer():
        n = max(round(n), 1)
        resize = (n**free_ranks) * output_variable_factor
    new_output = list(oupu)
    variables_sum = sum(variables)
    for i, v in zip(variables_index, variables):
        new_output[i] = int(v * n)
    return new_output


def collapse_at(list_, ind):
    if len(list_) > ind:
        return list_[:ind] + [np.prod(list_[ind:])]
    return list_


def resizing_args(inpu, oupu):
    oupu = dimensions_inference(inpu, oupu)
    if np.prod(inpu) != np.prod(oupu):
        for i, (inp, out) in enumerate(zip(inpu, oupu)):
            if inp != out:
                first_diff = i
                break
        else:
            first_diff = i + 1
        reshape_in = collapse_at([1] * max(2 - first_diff, 0) + inpu, 4)
        reshape_out = collapse_at([1] * max(2 - first_diff, 0) + oupu, 4)
        reshape_in = collapse_at(reshape_in, len(reshape_out) - 1)
        reshape_out = collapse_at(reshape_out, len(reshape_in) - 1)
        if len(reshape_in) == 3:
            pool = nn.AdaptiveAvgPool1d(reshape_out[2:])
        if len(reshape_in) == 4:
            pool = nn.AdaptiveAvgPool2d(reshape_out[2:])
        if len(reshape_in) == 5:
            pool = nn.AdaptiveAvgPool3d(reshape_out[2:])
        return [reshape_in, reshape_out[2:], oupu]
    else:
        return [oupu]


def resize(x, oupu):
    oupu = dimensions_inference(list(x.shape), oupu)
    args = resizing_args(list(x.shape), oupu)
    if len(args) == 3:
        if len(args[0]) == 3:
            pool = nn.AdaptiveAvgPool1d(args[1])
        if len(args[0]) == 4:
            pool = nn.AdaptiveAvgPool2d(args[1])
        if len(args[0]) == 5:
            pool = nn.AdaptiveAvgPool3d(args[1])
        x = x.view(args[0])
        x = pool(x)
    x = x.view(oupu)
    return x


x = rand([4, 63])
x = resize(x, oupu=[4, 1, -1, -1])
x.shape
