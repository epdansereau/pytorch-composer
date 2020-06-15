from .Layer import Layer

from .layers import Linear
from .layers import Conv2d
from .layers import MaxPool2d
from .layers import Reshape
from .layers import Flat
from .layers import Relu
from .layers import AdaptiveAvgPool1d
from .layers import AdaptiveAvgPool2d
from .layers import AdaptiveAvgPool3d
from .layers import RNN
from .layers import permute

# The classes for all the types of layers are saved in a dictionary. The key is the name of the classes.
# Example:
# layers["Linear"] will return the Linear class.
layers = {x.__name__: x for x in Layer.__subclasses__()}

def get_layer(layer_name):
    return layers[layer_name]