# Tests if the dimensions claimed by the comments of the generated code
# are the same as the real dimensions

from pytorch_composer.adaptive_resizing import resize
from torch import rand
import numpy as np

tests1 = [
    [[8,14],[8,14]],
    [[8,14],[8,2,7]],
    [[8],[1]],
    [[1],[8]],
    [[1],[1,8]],
    [[1,8],[1]],
]
tests2 = [
    [[4,64],[4,1,-1,-1]],
    [[4,63],[4,1,-1,-1]],
    [[1],[-3,-3,-50,-20]],
    [[43,55,12],[-1,-4,8]],
    [[43,55,12],[-1,-4,-8,5]],
    [[43,55,12],[-1,-4,8,-5]],
]
for test in tests1:
    x = rand(test[0])
    y = resize(x, test[1])
    print(test[1],list(y.shape))
    assert test[1] == list(y.shape)

for test in tests2:
    X = rand(test[0])
    Y = resize(X, test[1])
    x_neg = []
    y_neg = []
    print(test[1],list(Y.shape))
    for x,y in zip(test[1],list(Y.shape)):
        if x ==  abs(x):
            assert x == y
        else:
            x_neg.append(-x)
            y_neg.append(y)
    prod1 = np.prod(x_neg)
    prod2 = np.prod(x_neg)
    assert float(prod2/prod1).is_integer()
    

print("All tests passed")
    