from string import Template, Formatter
from collections import defaultdict

class CodeSection(Template):
    def __init__(self, template, settings, variables = None, imports = None):
        self._template = template
        if imports is None:
            self.imports = set()
        else:
            self.imports = set(imports)
        if variables is None:
            variables = {}
        self.variables = variables
        self.defaults = {"input_dim":None,"output_dim":None}
        self.__dict__ = {**self.defaults,**settings,**self.__dict__}

    @property
    def template(self):
        return self._template
        
    @property
    def args(self):
        return [x[1] for x in Formatter().parse(self.template) if x[1]]
        
    @property
    def settings(self):
        return {arg:defaultdict(str,self.__dict__)[arg] for arg in self.args}
    
    @staticmethod
    def write_imports(imports):
        code = ""
        for import_ in imports:
            if isinstance(import_, str):
                code += "import " + import_ + "\n"
            if isinstance(import_, tuple):
                code += "from " + import_[0] + " import " + import_[1] +"\n"
        return code
    
    @property
    def code(self):
        return Template(self.template).substitute(self.settings)
    
    @property
    def str_(self):
        return self.write_imports(self.imports) + self.code
    
    def __str__(self):
        return self.str_
    
    def __repr__(self):
        return self.str_
    
    @property
    def out(self):
        return {"x_dim":self.output_dim,
               "batch_rank":0}
    
    def set_output(self,output):
        raise ValueError("An output of {} was requested, but the CodeSection's output is {} and can't be changed".format(
                                                                                                 output,self.output_dim))
        
    def require_input(self, inputSection):
        return False
    
    def fit(self, inputSection):
        required_input = self.require_input(inputSection)
        if required_input:
            inputSection = inputSection.set_output(required_input)
        return inputSection
        