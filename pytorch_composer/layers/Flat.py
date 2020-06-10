from pytorch_composer.Layer import Layer


class Flat(Layer):

    def __init__(self, input_dim, batch_rank):
        self.layer_type = "flat"
        self.args = None
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.nn = None
        self.description = "Flatenning the data"
        self.batch_rank = batch_rank

        # Arguments:
        self.default_args = {}
        self.dimension_key = ""
        self.required_args = []
        self.kw_args = []

    @staticmethod
    def required_batch_rank(data_dim, data_rank, args):
        return 0  
    
    @classmethod
    def create(cls, input_dim, dimension_arg, other_args, batch_rank):
        return cls(input_dim, batch_rank)

    @staticmethod
    def valid_input_dims(input_dims, batch_rank):
        return Layer.change_rank(input_dims, 2, batch_rank)

    def update_block(self, block):
        # Nothing to do here since the reshape happens earlier
        return block
