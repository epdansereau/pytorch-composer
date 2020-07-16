from pytorch_composer.Layer import Layer


class Flat(Layer):

    def __init__(self, dimension_arg = None, other_args = None, variables = None):
        super().__init__(dimension_arg, other_args, variables, layer_type = "flat", description = "Flatenning the data")

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
