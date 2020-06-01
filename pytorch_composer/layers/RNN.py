from pytorch_composer.Layer import Layer


class RNN(Layer):

    def __init__(self, input_dim):
        self.layer_type = "rnn"
        self.args = None
        self.input_dim = input_dim
        self.output_dim = None
        self.nn = "nn.RNN"
        self.description = "Recurrent layer"

        #Arguments
        self.default_args = {
            "num_layers":1,
            "bias":True,
            "batch_first":False,
            "dropout":0.0,
            "bidirectional":False,
        }
        self.dimension_key = 'hidden_size'
        self.required_args = ['input_size', 'hidden_size']
        self.kw_args = ['num_layers', 'bias', 'batch_first', 'dropout', 'bidirectional']

    @classmethod
    def create(cls, input_dim, dimension_arg, other_args=None):
        if other_args is None:
            other_args = {}
        layer = cls(input_dim)
        args = layer.active_args(dimension_arg, other_args)
        args['input_size'] = input_dim[-1]
        args['hidden_size'] = dimension_arg
        layer.output_dim = input_dim.copy()
        layer.output_dim[-1] = dimension_arg
        layer.args = layer.write_args(args)
        return layer
    
    @staticmethod
    def valid_input_dims(input_dims):
        return Layer.change_rank(input_dims, 3)

    def update_block(self, block):
        block.hidden_var.append("h" + str(len(block.hidden_var) + 1))
        block.hidden_dims[block.hidden_var[-1]] = tuple([1] + self.output_dim[1:])
        return self.add_unique_layer(block, hidden = True)
