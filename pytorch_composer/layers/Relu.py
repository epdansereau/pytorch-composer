from pytorch_composer.Layer import Layer


class Relu(Layer):

    def __init__(self, variables):
        self.layer_type = "relu"
        self.args = None
        self.input_dim = variables.output_dim.copy()
        self.nn = None
        self.description = "Relu activation"
        self.variables = variables

        # Arguments:
        self.default_args = {}
        self.dimension_key = ""
        self.required_args = []
        self.kw_args = []

    # Main loop:

    # Valid permutation:

    # Valid input dimensions:

    # Creating the layer:

    # Updating the block object:

    def update_block(self, block):
        # Nothing to do here since the reshape happens earlier
        block.add_forward(["reshape", "x = F.relu(x)"])
