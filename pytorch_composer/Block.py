from collections import Counter, defaultdict


class Block():
    """A group of several layers"""

    def __init__(
            self,
            count=Counter(),
            groups=defaultdict(list),
            layers_list=[],
            forward_function=[]):
        self.count = count
        self.groups = groups
        self.layers_list = layers_list
        self.forward_function = forward_function

    def update(self, layer):
        return layer.update_block(self)

    def add_forward(self, line):
        self.forward_function.append(line)

    def add_layer(self, line):
        self.layers_list.append(line)
