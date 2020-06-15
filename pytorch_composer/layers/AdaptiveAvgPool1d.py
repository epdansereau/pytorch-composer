from pytorch_composer.Layer import Layer


class AdaptiveAvgPool1d(Layer):
    def __init__(self, variables):
        self.layer_type = "adaptiveavgpool1d"
        self.args = None
        self.input_dim = variables.output_dim.copy()
        self.nn = "nn.AdaptiveAvgPool1d"
        self.description = "Resizing with adaptive average pooling"
        self.variables = variables

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

    def get_valid_args(self, args):
        #TD: remove
        if isinstance(args["output_size"], tuple):
            args["output_size"] = args["output_size"][0]
        return args

    def update_variables(self, args):
        out = self.input_dim.copy()
        out[-1] = args["output_size"]
        self.variables.update_x(out)

    # Updating the block object:

    def update_block(self, block):
        self.add_reusable_layer(block)
