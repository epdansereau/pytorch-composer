from pytorch_composer.Layer import Layer


class AdaptiveAvgPool1d(Layer):
    def __init__(self, dimension_arg, other_args = None, variables = None):
        super().__init__(dimension_arg, other_args, variables)
        self.layer_type = "adaptiveavgpool1d"
        self.nn = "nn.AdaptiveAvgPool1d"
        self.description = "Resizing with adaptive average pooling"

        # Arguments:
        self.default_args = {
        }
        self.dimension_key = 'output_size'
        self.required_args = ['output_size']
        self.kw_args = []

    # Main loop:

    # Valid permutation:

    @staticmethod
    def required_batch_rank(data_dim, data_rank, args):
        return 0

    # Valid input dimensions:

    @staticmethod
    def valid_input_dims(input_dims, batch_rank):
        return Layer.change_rank(input_dims, 3, batch_rank)

    # Creating the layer:

    @property
    def valid_args(self):
        args = self.active_args
        if isinstance(args["output_size"], tuple):
            args["output_size"] = args["output_size"][0]
        return args

    def update_variables(self):
        out = self.input_dim.copy()
        out[-1] = self.valid_args["output_size"]
        self.variables.update_x(out)

    # Updating the block object:

    def update_block(self, block):
        self.add_reusable_layer(block)
