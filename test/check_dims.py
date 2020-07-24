# Tests if the dimensions claimed by the comments of the generated code
# are the same as the real dimensions

from pytorch_composer.loops import Loop
import pytorch_composer.datasets
import traceback

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


def test(sequence, long = False):
    ''' The accuracy should always be 100% '''
    if long:
        dataset = pytorch_composer.datasets.RandLongDataset({"shape":[29, 1, 11, 10]})
    else:
        dataset = pytorch_composer.datasets.RandDataset()
    model = pytorch_composer.Model(sequence, dataset)
    loop = Loop(model)
    debug_code = '''
test_result = {}

'''
    code = pytorch_composer.Code([dataset, model, loop])
    # adding test code:
    code[0].template = debug_code + code[0].template
    code[1].block.code = add_dims_check(code[1].block.code)
    print(code)
    test_result = code(returns = ["test_result"])

    correct = 0
    for line in test_result:
        if test_result[line][0] == test_result[line][1]:
            print(test_result[line])
            correct += 1
        else:
            print(test_result[line], "Mismatch!")
    print()
    result = f'{correct} / {len(test_result)}'
    print('accuracy:', result)
    return result


print("TEST1")
test1 = test(sequence1)
print()
print("TEST2")
test2 = test(sequence2, long = True)
print()
print("1:", test1)
print("2:", test2)
