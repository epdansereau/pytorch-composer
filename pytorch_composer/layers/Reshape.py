from pytorch_composer.Layer import Layer


class Reshape(Layer):
    def update_block(self, block):
        block.add_forward(["comment",
                           "{}: ".format(self.description),
                           tuple(self.input_dim),
                           " -> ",
                           tuple(self.output_dim)])
        if self.args == -1:
            block.add_forward(
                ["reshape", "x = x.view(-1,{})".format(self.output_dim[1])])
        else:
            block.add_forward(
                ["reshape", "x = x.view{}".format(tuple(self.output_dim))])
        return block
