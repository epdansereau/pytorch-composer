from string import Template, Formatter
from collections import defaultdict
import copy
import sys
import traceback
from traceback import TracebackException
import ast

class Variable:
    def __init__(self, name, dim, batch_rank = 0, vocab = None):
        self.name = name
        self.dim = dim
        self.batch_rank = batch_rank
        self.vocab = Vocab.create(vocab)
        
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
    
    def add_variable(self, type_, dim, batch_rank = 0, vocab = None):
        name = type_ + str(len(self[type_]))
        self[type_].append(Variable(name, dim, batch_rank, vocab))     
    
    def update_dim(self, type_, ind, new_dim):
        self[type_][ind].dim = new_dim
        
    def update_batch_rank(self, type_, ind, new_batch_rank):
        self[type_][ind].batch_rank = new_batch_rank
        
    def update_vocab(self, type_, ind, new_vocab):
        self[type_][ind].vocab = Vocab.create(new_vocab)
        
    def update_x(self,dim = None,batch_rank = None, vocab = None, ind = 0):
        if dim is not None:
            self.update_dim("x",ind,dim)
        if batch_rank is not None:
            self.update_batch_rank("x",ind,batch_rank)
        if vocab is not None:
            self.update_vocab("x",ind,vocab)   

class SettingsDict(dict):
    def __init__(self, dict_, linked_to = None):
        self.linked_to = linked_to
        super().__init__(dict_)
    
    def __repr__(self):
        # forcing the dict to be printed on several lines
        str_ = ',\n '.join(["{}: {}".format(k.__repr__(),v.__repr__()) for k,v in self.items()])
        return "{" + str_ + "}"
    
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.linked_to is not None:
            self.linked_to.update({key:value})
            
    def update(self, dict_ = None):
        if dict_ is None:
            dict_ = {}
        super().update(dict_)
        if self.linked_to is not None:
            self.linked_to.update(dict_)
            
    def unlink(self):
        self.linked_to = None
        
class Vocab:
    def __init__(self, classes = None, size = None, embed_dim = None, weights = None):
        self.size = size
        self.classes = classes
        if classes is not None and size is None:
            self.size = len(classes)
        self.embed_dim = embed_dim
        self.weights = weights
        
    @classmethod
    def create(cls, args):
        if isinstance(args,list):
            return cls(args)
        elif isinstance(args,cls) or args is None:
            return args
        elif isinstance(args,dict):
            return cls(**args)
        else:
            raise TypeError

class ComposerError(Exception):
    @staticmethod
    def _show_lines(selected_line, str_):
        '''Prints selected numbered lines from a string. '''
        all_lines = str_.splitlines() # empty string to start index at 1
        shown_lines = []
        lines_printed = min(5,len(all_lines))
        start_from = min(len(all_lines) - lines_printed, selected_line - 3)
        start_from = max(0, start_from)
        for line_number in range(start_from, start_from + lines_printed):
            if line_number + 1 == selected_line:
                sep = " -> "
            else:
                sep = "    "
            shown_lines.append(str(line_number + 1).rjust(3) + sep + all_lines[line_number].rstrip())       
        return "\n".join(shown_lines)
            
class CodeSection:
    def __init__(self,
                 variables = None,
                 settings = None,
                 defaults = None,
                 template = "",
                 imports = None,
                 linked_to = None,
                 returns = None,
                ):
        self._template = template
        if imports is None:
            self.imports = set()
        else:
            self.imports = set(imports)
        if settings is None:
            settings = {}
        if defaults is None:
            defaults = {}
        self.defaults = defaults
        self.entered_settings = settings
        self.set_variables(variables)
        self.linked_to = linked_to
        self.returns = returns
        
    def setitem(self, key, item):
        self.entered_settings[key] = item
        
    def __setitem__(self, key, item):
        self.setitem(key, item)
        self.update()

    def __getitem__(self, key):
        return self.settings[key]
    
    def __contains__(self, item):
        return item in self.settings
    
    def __str__(self):
        return self.str_
    
    def __repr__(self):
        return self.str_
    
    @staticmethod
    def execute(code_source, returns = None, env = None):
        if env is None:
            env = {}
        code_source = str(code_source)
        assert isinstance(code_source,str)
        parsed = ast.parse(code_source)
        code = compile(parsed, 'ComposerExec','exec')
        try:
            exec(code, env)
        except Exception as error:
            error_class = error.__class__.__name__
            if len(error.args):
                detail = error.args[0]
            else:
                detail = None
            _, _, tb_ = sys.exc_info()
            tb = traceback.extract_tb(tb_)
            trace_string = ""
            for frame in tb:
                trace_string += traceback.format_list([frame])[0]
                if frame.filename == 'ComposerExec':
                    line_number = frame.lineno
                    trace_string += "    " + code_source.splitlines()[line_number - 1] + "\n"
        else:
            if returns is None:
                return None
            else:
                variables = [env[v] for v in returns]
                if len(variables) == 1:
                    return variables[0]
                else:
                    return variables
        lines = ComposerError._show_lines(line_number,code_source)
        error_msg = f"\n{lines}\n{error_class} was raised at line {line_number} : {detail}\n\nTraceback:\n{trace_string}"
        raise ComposerError(error_msg)
        
    @property
    def template(self):
        return self._template
    
    @template.setter
    def template(self, template):
        self._template = template 
        
    @property
    def template_keys(self):
        return [x[1] for x in Formatter().parse(self.template) if x[1]]
    
    @property
    def settings(self):
        return SettingsDict({**self.defaults,**self.entered_settings}, self)
        
    @settings.setter
    def settings(self, settings):
        if not isinstance(settings, dict):
            raise TypeError
        self.update(settings)
    
    @property
    def active_settings(self):
        return self.settings
    
    @property
    def template_settings(self):
        return {arg:defaultdict(str,self.active_settings)[arg] for arg in self.template_keys}
    
    def unlink(self):
        self.linked_to = None
        
    def update(self, settings = None):
        if settings:
            keys = set(self.settings.keys())
            if keys.union(set(settings)) != keys:
                raise KeyError("Settings not found: {}".format(set(settings) - keys))
            self._update(settings)
        if self.linked_to is not None:
            self.linked_to.update()
            
    def _update(self, settings):
        # Updates only the section. Ignores keys not in defaults.
        for k, v in settings.items():
            if k in self.settings:
                self[k] = v
    
    def set_default_variables(self):
        pass
    
    def set_variables(self, variables):
        if variables is None:
            self.variables  = Vars({})
            self.set_default_variables()
        elif isinstance(variables, Vars):
            self.variables = variables.copy()
        elif isinstance(variables, CodeSection):
            self.variables = variables.variables.copy()
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
        self.set_variables(inputSection)
    
        