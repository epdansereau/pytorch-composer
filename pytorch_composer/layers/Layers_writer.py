class Layers_writer():
    '''
    Callable class that updates a Block object to add code from a new layer. Returns the updated Block object.
        
    args:
        block:
            The Block object.
        layer:
            The layer to be added.
    '''


    def __call__(self, block, layer):
        return getattr(self, layer.layer_type.lower())(block, layer)     

    def _add_unique_layer(self, block, layer):
        block.count[layer.layer_type] += 1
        ind = block.count[layer.layer_type]
        block.add_layer(["layer", "self.{}".format(layer.layer_type), ind, " = {}({})".format(layer.nn, layer.args)])
        block.add_forward(["comment",
                                 "{}: ".format(layer.description),
                                 tuple(layer.input_dim),
                                 " -> ",
                                 tuple(layer.output_dim)])
        block.add_forward(
            ["forward", "x = ", "self.{}{}".format(layer.layer_type, ind), "(x)"])
        return block

    def _add_reusable_layer(self, block, layer):
        is_new_group = not(layer.args in block.groups[layer.layer_type])
        if is_new_group:
            block.groups[layer.layer_type].append(layer.args)
        ind = block.groups[layer.layer_type].index(layer.args) + 1
        if is_new_group:
            block.add_layer(["layer", "self.{}".format(
                layer.layer_type), ind, " = {}({})".format(layer.nn, layer.args)])
        block.add_forward(["comment",
                                 "Reshaping the data: ",
                                 tuple(layer.input_dim),
                                 " -> ",
                                 tuple(layer.output_dim)])
        block.add_forward(
            ["forward", "x = ", "self.{}{}".format(layer.layer_type, ind), "(x)"])
        return block

    def conv2d(self, block, layer):
        return self._add_unique_layer(block, layer)

    def maxpool2d(self, block, layer):
        return self._add_reusable_layer(block,layer)

    def linear(self, block, layer):
        return self._add_unique_layer(block, layer)

    def flat(self, block, layer):
        # Nothing to do here since the reshape happens earlier
        return block

    def reshape(self, block, layer):
        block.add_forward(["comment",
                                 "{}: ".format(layer.description),
                                 tuple(layer.input_dim),
                                 " -> ",
                                 tuple(layer.output_dim)])
        if layer.args == -1:
            block.add_forward(
                ["reshape", "x = x.view(-1,{})".format(layer.output_dim[1])])
        else:
            block.add_forward(
                ["reshape", "x = x.view{}".format(tuple(layer.output_dim))])
        return block

    def relu(self, block, layer):
        block.add_forward(["reshape", "x = F.relu(x)"])
        return block


write = Layers_writer()
