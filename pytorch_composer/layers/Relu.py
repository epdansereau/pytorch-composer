from pytorch_composer.Layer import Layer


class Relu(Layer):

    def __init__(self, dimension_arg = None, other_args = None, variables = None):
        super().__init__(dimension_arg, other_args, variables, layer_type = "relu", description = "Relu activation")

    # Main loop:

    # Valid permutation:

    # Valid input dimensions:

    # Creating the layer:

    # Updating the block object:

    def update_block(self, block):
        # Nothing to do here since the reshape happens earlier
        block.add_forward(["reshape", "x = F.relu(x)"])
