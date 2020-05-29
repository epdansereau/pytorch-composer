from torch import nn, rand
import numpy as np
import math

'''
We take a tensor of any shape and convert it to a tensor of any other shape, using nn.view and adaptive pooling layers.
'''


def dimensions_inference(inpu, oupu):
    '''
    input: inpu(list of int), oupu(list of int, can contain negative numbers).
    output: new_output(list of int)

    Usually, nn.view does inference on a single dimension:
        (4,64) -> (4,8,-1) : (4,8,8)
    Here, we allow inference on multiple dimensions:
        (4,64) -> (4,1,-1,-1) : (4,1,8,8)
    The negative integers don't have to be -1, they can be any negative ints to represent different proportions:
        (4,54) -> (4,1,-2,-3) -> (4,1,6,9)
    If the equation for inference can't be solved using integers, we give the result for the nearest input size
    that has a solution, and an adaptive layer will have to be used to make the input size fit.
        (4,63) -> (4,1,-1,-1) : (4,1,8,8)

    '''
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
    n = max(round(n), 1)
    new_output = list(oupu)
    for i, v in zip(variables_index, variables):
        new_output[i] = int(v * n)
    return new_output


def collapse_at(list_, ind):
    if len(list_) > ind:
        return list_[:ind] + [np.prod(list_[ind:])]
    return list_


def resizing_args(inpu, oupu):
    '''
    input: inpu(list of int), oupu(list of int, can contain negative numbers).
    output:
        If a pooling layer is not necessary:
            A list of lenght 1, that contains a list of arguments for an nn.view layer.
        If a pooling layer is necessary:
            A list of lenght 3, that contains a arguments for a first nn.view layer, arguments for
            an adaptive pooling layer, arguments for a second nn.view layer. The pooling function to
            be used should be 1d if the pooling arguments are lenght one, 2d if the pooling arguments
            are lenght 2, etc.

    Provides arguments to convert a tensor of any shape into any shape. Inference on multiple dimensions is allowed.

    '''
    oupu = dimensions_inference(inpu, oupu)
    if np.prod(inpu) != np.prod(oupu):
        for i, (inp, out) in enumerate(zip(inpu, oupu)):
            if inp != out:
                first_diff = i
                break
        else:
            first_diff = i
        reshape_in = collapse_at([1] * max(2 - first_diff, 0) + inpu, 4)
        reshape_out = collapse_at([1] * max(2 - first_diff, 0) + oupu, 4)
        reshape_in = collapse_at(reshape_in, len(reshape_out) - 1)
        reshape_out = collapse_at(reshape_out, len(reshape_in) - 1)
        return [reshape_in, reshape_out[2:], oupu]
    else:
        return [oupu]


def resize(x, oupu):
    # Converts a tensor into any shape, using adaptive pooling layers if needed.
    # inputs:
    #     x: a tensor, oupu: any shape(list, inference on multiple dimensions is allowed.)
    # output
    #     A new tensor of the desired shape.
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
    x = x.view(args[-1])
    return x
