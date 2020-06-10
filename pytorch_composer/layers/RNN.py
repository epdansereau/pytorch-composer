from pytorch_composer.Layer import Layer


class RNN(Layer):

    def __init__(self, input_dim, batch_rank):
        self.layer_type = "rnn"
        self.args = None
        self.input_dim = input_dim
        self.output_dim = None
        self.nn = "nn.RNN"
        self.description = "Recurrent layer"
        self.batch_rank = batch_rank

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
    
    @staticmethod
    def required_batch_rank(data_dim, data_rank, args):
        if "batch_first" in args:
            return int(not(args["batch_first"]))
        return 1
        
        
    @classmethod
    def create(cls, input_dim, dimension_arg, other_args, batch_rank):
        if other_args is None:
            other_args = {}
        layer = cls(input_dim, batch_rank)
        args = layer.active_args(dimension_arg, other_args)
        args['input_size'] = input_dim[-1]
        args['hidden_size'] = dimension_arg
        layer.output_dim = input_dim.copy()
        layer.output_dim[-1] = dimension_arg
        if args["batch_first"]:
            layer.hidden_dim = tuple([1, layer.output_dim[0], layer.output_dim[2]])
        else:
            layer.hidden_dim = tuple([1] + layer.output_dim[1:])
        layer.args = layer.write_args(args)
        return layer
    
    @staticmethod
    def valid_input_dims(input_dims, batch_rank):
        return Layer.change_rank(input_dims, 3, batch_rank)

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
        ind = str(len(block.hidden))
        hidden_var = ("h" + ind, self.hidden_dim, self.batch_rank)
        block.hidden.append(hidden_var)
        return self.add_hidden_layer(block, hidden_var[0], ind)
    
    
    def add_hidden_layer(self, block, hidden_var, ind):
        # Updates the block when the layer should not be reused in the forward function (i.e. when the
        # layer has weights).
        block.count[self.layer_type] += 1
        block.add_layer(["layer", "self.{}".format(
            self.layer_type), ind, " = {}({})".format(self.nn, self.args)])
        block.add_forward(["comment",
                           "{}: ".format(self.description),
                           tuple(self.input_dim),
                           " -> ",
                           tuple(self.output_dim)])
        block.add_forward(
            ["forward", "x, {} = ".format(hidden_var),
             "self.{}{}".format(self.layer_type, ind),
             "(x, {})".format(hidden_var)])

        return block
