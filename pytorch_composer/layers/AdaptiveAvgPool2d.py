from pytorch_composer.Layer import Layer


class AdaptiveAvgPool2d(Layer):

    def __init__(self, input_dim, batch_rank):
        self.layer_type = "adaptiveavgpool2d"
        self.args = None
        self.input_dim = input_dim
        self.output_dim = None
        self.nn = "nn.AdaptiveAvgPool2d"
        self.description = "Resizing with adaptive average pooling"
        self.batch_rank = batch_rank

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
        return Layer.change_rank(input_dims, 4, batch_rank)

    # Creating the layer:

    def get_valid_args(self, args):
        args["output_size"] = self.int_to_tuple(args["output_size"])
        return args

    def get_output_dim(self, args):
        out = self.input_dim.copy()
        out[-2] = args["output_size"][0]
        out[-1] = args["output_size"][1]
        return out

    # Updating the block object:

    def update_block(self, block):
        return self.add_reusable_layer(block)
