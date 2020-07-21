from pytorch_composer.Layer import Layer


class Embedding(Layer):

    def __init__(self, dimension_arg, other_args = None, variables = None):
        self.vocab = variables["x"][0].vocab
        if self.vocab is None:
            raise ValueError("Not supported: the embedding layer has to receive a tensor of int64")
        self.from_pretrained = self.vocab.weights is not None

        if self.from_pretrained:
            ''' Embeddings from pretrained '''
            super().__init__(
                     dimension_arg,
                     other_args,
                     variables,
                     layer_type = "embedding",
                     nn = "nn.Embedding.from_pretrained",
                     description = "Embedding layer",
                     default_args = {
                        'embeddings':None,
                        "freeze":True,
                        "padding_idx":None,
                        "max_norm":None,
                        "norm_type":2.0,
                        "scale_grad_by_freq":False,
                        "sparse":False,
                     },
                     dimension_key = "embeddings",
                     required_args = ['embeddings'],
                     kw_args = ['freeze', 'padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse'],
                     spaces = {
                        "out_channels":"n",
                     }
            )
        else:
            ''' New embeddings '''
            super().__init__(
                     dimension_arg,
                     other_args,
                     variables,
                     layer_type = "embedding",
                     nn = "nn.Embedding",
                     description = "Embedding layer",
                     default_args = {
                        "padding_idx":None,
                        "max_norm":None,
                        "norm_type":2.0,
                        "scale_grad_by_freq":False,
                        "sparse":False,
                        "_weight":None,
                        'embedding_dim':100, #default size
                     },
                     dimension_key = 'embedding_dim',
                     required_args = ['num_embeddings', 'embedding_dim'],
                     kw_args = ['padding_idx', 'max_norm', 'norm_type', 'scale_grad_by_freq', 'sparse', '_weight'],
                     spaces = {}
             )


    @property
    def valid_args(self):
        args = self.active_args
        if self.from_pretrained:
            args['embeddings'] = self.vocab.weights
        else:
            args['num_embeddings'] = self.vocab.size
        return args
        
    def update_model(self, model):
        out = self.input_dim.copy()
        if self.from_pretrained:
            out += [self.vocab.embed_dim]
        else:
            out += [self.valid_args['embedding_dim']]
        model.block.variables.update_x(out,vocab = None)
        
        self.add_unique_layer(model.block)
        