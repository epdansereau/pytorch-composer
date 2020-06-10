from pytorch_composer.Layer import Layer


class Relu(Layer):

    def __init__(self, input_dim, batch_rank):
        self.layer_type = "relu"
        self.args = None
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.nn = None
        self.description = "Relu activation"
        self.batch_rank = batch_rank

        # Arguments:
        self.default_args = {}
        self.dimension_key = ""
        self.required_args = []
        self.kw_args = []

    @classmethod
    def create(cls, input_dim, dimension_arg, other_args, batch_rank):
        return cls(input_dim, batch_rank)

    def update_block(self, block):
        # Nothing to do here since the reshape happens earlier
        block.add_forward(["reshape", "x = F.relu(x)"])
        return block
