from pytorch_composer.Layer import Layer


class Embedding(Layer):

    def __init__(self, dimension_arg, other_args = None, variables = None):
        self.vocab = variables["x"][0].vocab
        if self.vocab is None:
            raise ValueError("Not supported: the embedding layer has to receive a tensor of int64")
        self.from_pretrained = self.vocab.weights is not None

        super().__init__(
                 dimension_arg,
                 other_args,
                 variables,
                 layer_type = "embedding",
                 description = "Embedding layer",
        )
        
        if self.from_pretrained:
            self.pretrained_embeds()
        else:
            self.new_embeds()

    def pretrained_embeds(self):
        self.nn = "nn.Embedding.from_pretrained"
        self.default_args = {
            'embeddings':None,
            "freeze":True,
            "padding_idx":None,
            "max_norm":None,
            "norm_type":2.0,
            "scale_grad_by_freq":False,
            "sparse":False,
        }
        self.dimension_key = "embeddings"
        self.required_args = ['embeddings']
        self.kw_args = ['freeze', 'padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse']
        self.spaces = {
            "out_channels":"n",
        }
        
    def new_embeds(self):
        self.nn = "nn.Embedding"
        self.default_args = {
            "padding_idx":None,
            "max_norm":None,
            "norm_type":2.0,
            "scale_grad_by_freq":False,
            "sparse":False,
            "_weight":None,
            'embedding_dim':100, #default size
        }
        self.dimension_key = "embedding_dim"
        self.required_args = ['num_embeddings', 'embedding_dim']
        self.kw_args = ['padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse', '_weight']
        self.spaces = {}        
            
    @property
    def valid_args(self):
        args = self.active_args
        if self.from_pretrained:
            args['embeddings'] = self.vocab.weights
        else:
            args['num_embeddings'] = self.vocab.size
        return args
        
    def update_model(self, model):
        input_dim = model.block.output_dim.copy()
        out = input_dim.copy()
        if self.from_pretrained:
            out += [self.vocab.embed_dim]
        else:
            out += [self.valid_args['embedding_dim']]
        model.block.variables.update_x(out,vocab = None)
        
        self.add_unique_layer(model.block)
        