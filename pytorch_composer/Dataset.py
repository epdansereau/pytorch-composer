from pytorch_composer import CodeSection
from pytorch_composer import Code


class Dataset(CodeSection):
        

    def __call__(self):
        self._trainloader, self._testloader = self.execute(str(self), returns = ["trainloader","testloader"])        
        
    @property
    def trainloader(self):
        if self._trainloader is not None:
            return self._trainloader
        raise RuntimeError("The trainloader hasn't been loaded. Call the dataset to load it.")
    
    @property
    def testloader(self):
        try:
            return self.testloader
        except AttributeError:
            pass
        raise RuntimeError("The testloader hasn't been loaded. Call the dataset to load it.")
        
    def get_batch(self):
        return next(iter(self.trainloader))
        