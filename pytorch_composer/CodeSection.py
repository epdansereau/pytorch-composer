from string import Template, Formatter
from collections import defaultdict

class CodeSection(Template):
    def __init__(self, template, settings, input_dim = None, output_dim = None, defaults = None):
        super().__init__(template)
        if defaults is None:
            defaults = {}
        self.defaults = defaults
        self.__dict__ = {**self.__dict__,**settings,**self.defaults}
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    @property
    def args(self):
        return [x[1] for x in Formatter().parse(self.template) if x[1]]
        
    @property
    def settings(self):
        return {arg:defaultdict(str,self.__dict__)[arg] for arg in self.args}
        
    def __str__(self):
        return self.substitute(self.settings)
    
    def __repr__(self):
        return self.substitute(self.settings)