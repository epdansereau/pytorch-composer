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
        
        self.hidden_dim = None
        
        self.batch_rank_ = 1

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
        if args["batch_first"]:
            layer.hidden_dim = tuple([1, layer.output_dim[0], layer.output_dim[2]])
        else:
            layer.hidden_dim = tuple([1] + layer.output_dim[1:])
        if args['batch_first']:
            layer.batch_rank_ = 0
        layer.args = layer.write_args(args)
        return layer
    
    @staticmethod
    def valid_input_dims(input_dims):
        if len(input_dims) !=3:
            return [4,-1,-1]
        
    def batch_rank(self):
        return self.batch_rank_

    @staticmethod
    def valid_permutation(data_dim, data_rank, args = None):
        batch_rank = 1
        if args:
            if "batch_first" in args:
                batch_rank = int(not(args["batch_first"]))
            
        if data_rank == batch_rank:
            return False
        else:
            perm = [i for i in range(max(len(data_dim),batch_rank +1))]
            perm = perm[:data_rank] + perm[data_rank+1:]
            perm = perm[:batch_rank] + [data_rank]+ perm[batch_rank:]
            return perm

    def update_block(self, block):
        block.hidden_var.append("h" + str(len(block.hidden_var) + 1))
        block.hidden_dims[block.hidden_var[-1]] = self.hidden_dim
        return self.add_unique_layer(block, hidden = True)
