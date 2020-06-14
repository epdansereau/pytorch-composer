from string import Template, Formatter
from collections import defaultdict
import copy

class Variable:
    def __init__(self, name, dim, batch_rank):
        self.name = name
        self.dim = dim
        self.batch_rank = batch_rank
        
    def __repr__(self):
        return str(tuple([self.name,self.dim,self.batch_rank]))
    
class Vars:

    def __init__(self, vars_):
        self.vars = defaultdict(list, vars_)
        
    def __setitem__(self, key, item):
        self.vars[key] = item

    def __getitem__(self, key):
        return self.vars[key]
    
    def __contains__(self, item):
        return item in self.vars

    def __repr__(self):
        return "Vars({})".format("\n     ".join([str(key) + ":" + str(val) for key, val in self.vars.items()]))
        
    def names(self, type_):
        return [v.name for v in self[type_]]
    
    @property
    def output_dim(self):
        return self["x"][0].dim
    
    @property
    def batch_rank(self):
        return self["x"][0].batch_rank
    
    def copy(self):
        return copy.deepcopy(self)
    
    # Common operations:
    
    def add_variable(self, type_, dim, batch_rank):
        name = type_ + str(len(self[type_]))
        self[type_].append(Variable(name, dim, batch_rank))     
    
    def update_dim(self, type_, ind, new_dim):
        self[type_][ind].dim = new_dim
        
    def update_batch_rank(self, type_, ind, new_batch_rank):
        self[type_][ind].batch_rank = new_batch_rank
        
    def update_x(self,new_dim = None,new_batch_rank = None, ind = 0):
        if new_dim is not None:
            self.update_dim("x",ind,new_dim)
        if new_batch_rank is not None:
            self.update_batch_rank("x",ind,new_batch_rank)   
    

class CodeSection(Template):
    def __init__(self, variables = None, template = "", entered_settings = None, imports = None):
        self._template = template
        if imports is None:
            self.imports = set()
        else:
            self.imports = set(imports)
        self.set_variables(variables)
        if entered_settings is None:
            entered_settings = {}
        self.defaults = {}
        self.entered_settings = entered_settings
        
    def __setitem__(self, key, item):
        self.entered_settings[key] = item

    def __getitem__(self, key):
        return self.settings[key]
    
    def __contains__(self, item):
        return item in self.settings
    
    def __str__(self):
        return self.str_
    
    def __repr__(self):
        return self.str_
        
    @property
    def template(self):
        return self._template
        
    @property
    def template_keys(self):
        return [x[1] for x in Formatter().parse(self.template) if x[1]]
    
    @property
    def settings(self):
        return {**self.defaults,**self.entered_settings}
        
    @property
    def active_settings(self):
        return self.settings
    
    @property
    def template_settings(self):
        return {arg:defaultdict(str,self.active_settings)[arg] for arg in self.template_keys}
    
    def set_default_variables(self):
        pass
    
    def set_variables(self, variables):
        if variables is None:
            self.variables  = Vars({})
            self.set_default_variables()
        elif isinstance(variables, Vars):
            self.variables = variables
        elif isinstance(variables, CodeSection):
            self.variables = variables.variables
        else:
            raise ValueError
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
        return Template(self.template).substitute(self.template_settings)
    
    @property
    def str_(self):
        return self.write_imports(self.imports) + self.code
    
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
    
        