from pytorch_composer.Layer import Layer


class Flat(Layer):

    def __init__(self, variables):
        self.layer_type = "flat"
        self.args = None
        self.input_dim = variables.output_dim.copy()
        self.nn = None
        self.description = "Flatenning the data"
        self.variables = variables

        # Arguments:
        self.default_args = {}
        self.dimension_key = ""
        self.required_args = []
        self.kw_args = []

    # Main loop:

    # Valid permutation:

    @staticmethod
    def required_batch_rank(data_dim, data_rank, args):
        return 0

    # Valid input dimensions:

    @staticmethod
    def valid_input_dims(input_dims, batch_rank):
        return Layer.change_rank(input_dims, 2, batch_rank)

    # Creating the layer:

    # Updating the block object:
        # Nothing to do here since the reshape happens earlier in the loop.
