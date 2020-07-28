# Tests if the dimensions claimed by the comments of the generated code
# are the same as the real dimensions

from pytorch_composer.loops import Loop
import pytorch_composer.datasets
import traceback

from layer_lab import rand_layers

sequence1 = [
    ["Conv2d", 6],
    ["MaxPool2d", 2],
    ["Linear", 52],
    ["Relu"],
    ["MaxPool2d", 2],
    ["Linear", 43],
    ["MaxPool2d", 2],
    ["Conv2d", 65],
    ["MaxPool2d", 2],
    ["Conv2d", 47],
    ["Flat"],
    ["Linear", 52],
    ["Relu"],
    ["Linear", 38],
    ["Conv2d", 8],
    ["MaxPool2d", 2],
    ["Conv2d", 47],
    ["Flat"],
    ["MaxPool2d", 2],
    ["Linear", 12],
    ["AdaptiveAvgPool1d", 122],
    ["Linear", 53],
    ["Conv2d", 65],
    ["AdaptiveAvgPool2d", 100],
]
sequence2 = [
    ['RNN', 29, {'input_size': 18, 'hidden_size': 30}],
    ["MaxPool2d", 2],
    ["Linear", 52],
    ["MaxPool2d", 2],
    ["RNN", 24],
    ["MaxPool2d", 2],
    ["Conv2d", 65],
    ["MaxPool2d", 2],
    ["Conv2d", 12],
    ["RNN", 24],
    ["Relu"],
]


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


def number_lines(code):
    with_number = ""
    for n, line in enumerate(str(code).split("\n")):
        with_number += str(n + 1).zfill(4) + ":" + line + "\n"
    return with_number


def test_dims(sequence, datatype = "float", verbose = True):
    ''' The accuracy should always be 100% '''
    if verbose:
        print(sequence)
    
    if datatype == "float":
        dataset = pytorch_composer.datasets.RandDataset()
    elif datatype == "int":
        dataset = pytorch_composer.datasets.RandLongDataset()
    elif datatype == "pretrained":
        dataset = pytorch_composer.datasets.RandPretrainedDataset()
    model = pytorch_composer.Model(sequence, dataset)
    loop = Loop(model)
    debug_code = '''
test_result = {}

'''
    code = pytorch_composer.Code([dataset, model, loop])
    # adding test code:
    code[0].template = debug_code + code[0].template
    code[1].block.code = add_dims_check(code[1].block.code)
    if verbose:
        print(code)
    test_result = code(returns = ["test_result"])

    correct = 0
    for line in test_result:
        if test_result[line][0] == test_result[line][1]:
            if verbose:
                print(test_result[line])
            correct += 1
        else:
            if verbose:
                print(test_result[line], "Mismatch!")
    result = f'{correct} / {len(test_result)}'
    print('accuracy:', result)
    assert correct == len(test_result)
    return result

def check_dims():    
    print("TEST1")
    for _ in range(50):
        test_dims(rand_layers())
    print()
    print("TEST2")
    for _ in range(50):
        test_dims(rand_layers(), datatype = "int")
    print()
    print("TEST3")
    for _ in range(50):
        test_dims(rand_layers(), datatype = "pretrained")
    print()

if __name__ == "__main__":
    check_dims()
