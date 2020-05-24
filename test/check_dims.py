# Tests if the dimensions claimed by the comments of the generated code
# are the same the real dimensions

import pytorch_composer
import torch

input_dim = [5, 2, 32, 12]
sequence = [
    ["conv2d", 6],
    ["maxpool2d", 2],
    ["linear", 52],
    ["relu"],
    ["maxpool2d", 2],
    ["linear", 43],
    ["maxpool2d", 2],
    ["conv2d", 65],
    ["maxpool2d", 2],
    ["conv2d", 47],
    ["flat"],
    ["linear", 52],
    ["relu"],
    ["linear", 38],
    ["conv2d", 8],
    ["maxpool2d", 2],
    ["conv2d", 47],
    ["flat"],
    ["maxpool2d", 2],
    ["linear", 12],
]

executable = '''
from torch import nn, rand
import torch.nn.functional as F
global test_result
test_result = {{}}
{}
model = Net()
data = rand{}
model(data)
'''


def add_dims_check(code):
    '''Adds a test after each comment about dimensions, and after the next forward function that changes
       the dimensions when there is one.
    '''
    new_code = []
    checking_for_output = False
    output_claimed = tuple()
    for i, line in enumerate(code):
        if line[0] == "comment" and len(line) == 3:
            new_code.append(line)
            if isinstance(line[2], tuple):
                template = 'test_result[{}] = [{}, tuple(x.size())]'.format(
                    i, line[2])
                new_code.append(["code", template])
        elif line[0] == "comment" and len(line) == 5:
            new_code.append(line)
            if isinstance(line[2], tuple) and isinstance(line[4], tuple):
                template = 'test_result[{}] = [{}, tuple(x.size())]'.format(
                    i, line[2])
                new_code.append(["code", template])
                checking_for_output = True
                output_claimed = line[4]
        elif line[0] == "forward" and checking_for_output:
            checking_for_output = False
            new_code.append(line)
            template = 'test_result[{}] = [{}, tuple(x.size())]'.format(
                i, output_claimed)
            new_code.append(["code", template])
        else:
            new_code.append(line)
    return new_code


def test(input_dim, sequence):
    ''' The accuracy should always be 100% '''
    model_code = pytorch_composer.write_model(input_dim, sequence)
    model_code.code_text = add_dims_check(model_code.code_text)
    model_code = model_code.formatted()
    model_code = executable.format(model_code, tuple(input_dim))
    exec(model_code, globals(), globals())
    # passing test results from globals
    correct = 0
    for line in test_result:
        if test_result[line][0] == test_result[line][1]:
            print(test_result[line])
            correct += 1
        else:
            print(test_result[line], "Mismatch!")
    print()
    print('accuracy:', f'{correct} / {len(test_result)}')


test(input_dim, sequence)
