from pytorch_composer.CodeSection import CodeSection

ROOT = "Path.home() / '.pyt-comp-data'"
NUM_WORKERS = 0

# == Vision datasets ==

class CIFAR10(CodeSection):
    def __init__(self, settings = None):
        template = '''
# Load and normalize the dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
     
root = ${root}
if not os.path.exists(root):
    os.makedirs(root)

trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=${batch_size},
                                          shuffle=True, num_workers=${num_workers})

testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=${batch_size},
                                         shuffle=False, num_workers=${num_workers})
'''
        defaults = {
            "batch_size":4,
            "num_workers":NUM_WORKERS,
            "root":ROOT
        }
        
        imports = set((
            "torch",
            "torchvision",
            "torchvision.transforms as transforms",
            ("pathlib", "Path"),
            "os",
        ))
        super().__init__(None, settings, defaults, template, imports)
        
    def setitem(self, key, item):
        self.entered_settings[key] = item
        if key == "batch_size":
            self.set_variables(None)
    
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",[self["batch_size"],3,32,32],0)
        self.variables.add_variable("y",[self["batch_size"]],0,10)
        
class MNIST(CodeSection):
    def __init__(self, settings = None):
        template = '''
# Load and normalize the dataset
transform = transforms.Compose(
    [transforms.ToTensor()])
    
root = ${root}
if not os.path.exists(root):
    os.makedirs(root)

trainset = torchvision.datasets.MNIST(root=root, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=${batch_size},
                                          shuffle=True, num_workers=${num_workers})

testset = torchvision.datasets.MNIST(root=root, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=${batch_size},
                                         shuffle=False, num_workers=${num_workers})
'''
        defaults = {
            "batch_size":4,
            "num_workers":NUM_WORKERS,
            "root":ROOT
        }
        
        imports = set((
            "torch",
            "torchvision",
            "torchvision.transforms as transforms",
            ("pathlib", "Path"),
            "os",
        ))
        super().__init__(None, settings, defaults, template, imports)
        
    def setitem(self, key, item):
        self.entered_settings[key] = item
        if key == "batch_size":
            self.set_variables(None)
    
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",[self["batch_size"],1,28,28],0)
        self.variables.add_variable("y",[self["batch_size"]],0,10)
        
# == Text dataset ==

# List of available embeddings
embeddings = ["charngram.100d",
"fasttext.en.300d",
"fasttext.simple.300d",
"glove.42B.300d",
"glove.840B.300d",
"glove.twitter.27B.25d",
"glove.twitter.27B.50d",
"glove.twitter.27B.100d",
"glove.twitter.27B.200d",
"glove.6B.50d",
"glove.6B.100d",
"glove.6B.200d",
"glove.6B.300d",]

def get_emb_dim(embedding):
    return int(embedding.split(".")[-1][:-1])

class _AG_NEWS(CodeSection):
    def __init__(self, settings = None):
        template = '''
        
root = ${root}
if not os.path.exists(root):
    os.makedirs(root)

dataset_name = "AG_NEWS"
dataset_tar = download_from_url(URLS[dataset_name], root=root)

def csv_iterator(_path):
    with io.open(_path, encoding="utf8") as f:
        reader = unicode_csv_reader(f)
        for row in reader:
            yield ' '.join(row[1:]), row[0]
            
def data_from_iterator(_path, fields):
    examples = []
    with tqdm(unit_scale=0, unit=' lines') as t:
        for text, label in csv_iterator(_path):
            examples.append(data.Example.fromlist(
                            [text, label], fields))
            t.update(1)
    return examples

class BucketWrapper(data.BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            yield batch.text, batch.label
            
TEXT = data.Field(sequential=True, tokenize='spacy', lower=True, fix_length = ${sequence_length})
LABEL = data.Field(sequential=False, unk_token = None)
fields = [("text",TEXT),("label",LABEL)]
            
train_data_path = root / "train_{}.data".format(dataset_name)
test_data_path = root / "test_{}.data".format(dataset_name)

if train_data_path.exists() and test_data_path.exists():
    train_data = data.Dataset(torch.load(train_data_path), fields)
    test_data = data.Dataset(torch.load(test_data_path), fields)
else:
    extracted_files = extract_archive(dataset_tar)
    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    train_path = root / train_csv_path
    test_path = root / test_csv_path

    train_data = data_from_iterator(train_path, fields)
    print("Saving train data to {}".format(train_data_path))
    torch.save(train_data, train_data_path)
    train_data = data.Dataset(train_data, fields)
    
    test_data = data_from_iterator(test_path, fields)
    print("Saving test data to {}".format(test_data_path))
    torch.save(test_data, test_data_path)
    test_data = data.Dataset(test_data, fields)
            
TEXT.build_vocab(train_data, vectors="${embedding}")
LABEL.build_vocab(train_data)
        
trainloader = BucketWrapper(train_data,${batch_size})
testloader = BucketWrapper(test_data,${batch_size})
'''
        defaults = {
            "batch_size":4,
            "root":ROOT,
            "sequence_length":150,
            "classes":4,
            "embedding":"glove.6B.100d",
        }
        
        imports = set((
            "io",
            ("tqdm", "tqdm"),
            ("pathlib", "Path"),
            "os",
            ("torchtext", "data, datasets"),
            ("torchtext.datasets.text_classification", "URLS"),
            ("torchtext.utils", "download_from_url, extract_archive, unicode_csv_reader"),
            "torch",
            ("torch","nn"),
        ))
        super().__init__(None, settings, defaults, template, imports)
        
    def set_variables(self, _):
        super().set_variables(None)
        embed_dim = get_emb_dim(self["embedding"])
        self.variables.add_variable("x",
                                    [self["sequence_length"], self["batch_size"]],
                                    1,
                                    {"embed_dim":embed_dim,
                                     "weights":"TEXT.vocab.vectors"}
                                   )
        self.variables.add_variable("y",[self["batch_size"]],0,self["classes"])
        
class AG_NEWS(CodeSection):
    #to be replaced soon
    def __init__(self, settings = None):
        template = '''
        
root = ${root}
if not os.path.exists(root):
    os.makedirs(root)

train_data_path = root / "AG_NEWS_trainNgram1.data"
test_data_path = root / "AG_NEWS_testNgram1.data"

if train_data_path.exists() and test_data_path.exists():
    trainset = torch.load(train_data_path)
    testset = torch.load(test_data_path)
else:
    trainset, testset = text_classification.DATASETS["AG_NEWS"](root=root)
    print("Saving train data to {}".format(train_data_path))
    torch.save(trainset, train_data_path)
    print("Saving test data to {}".format(test_data_path))
    torch.save(testset, test_data_path)

def generate_batch(batch):
    sequence_length = ${sequence_length}
    pad_token = 0
    
    input_ids = []
    labels = []
    
    for sequence in batch:
        input_id = sequence[1][:sequence_length]
        input_id = torch.cat((input_id, torch.zeros(sequence_length - len(input_id),dtype = torch.int64)))
        input_ids.append(input_id)
        labels.append(sequence[0])
    return torch.stack(input_ids).long(), torch.IntTensor(labels).long()
    
trainloader = DataLoader(
    trainset,
    batch_size=${batch_size},
    collate_fn=generate_batch,
    pin_memory=True)
    
testloader = DataLoader(
    testset,
    batch_size=${batch_size},
    collate_fn=generate_batch,
    pin_memory=True)    
'''
        defaults = {
            "batch_size":4,
            "root":ROOT,
            "sequence_length":200,
            "classes":4,
        }
        
        imports = set((
            "torch",
            ("torchtext.datasets","text_classification"),
            ("torch.utils.data","DataLoader"),
            ("pathlib", "Path"),
            "os",
        ))
        self.vocab_size = 95812
        super().__init__(None, settings, defaults, template, imports)
        
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",[self["batch_size"],self["sequence_length"]],0,self.vocab_size)
        self.variables.add_variable("y",[self["batch_size"]],0,self["classes"])

# == Debugging datasets ==

class RandDataset(CodeSection):
    def __init__(self, settings = None):
        template = '''
# Tensor with random values
class Rand:
    def __init__(self):
        self.shape = ${shape}
        
    def __getitem__(self, i):
        return torch.rand(self.shape), torch.randint(${classes},[1])[0]
        
    def __len__(self):
        return ${len}


trainset = Rand()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=${batch_size}, shuffle=False,
                                                        num_workers=0)

testset = Rand()

testloader = torch.utils.data.DataLoader(testset, batch_size=${batch_size}, shuffle=False,
                                        num_workers=0)
'''
        defaults = {
            "shape":[3,32,32],
            "batches":2,
            "batch_size":4,
            "classes":10,
        }
        
        imports = set((
            "torch",
        ))
        super().__init__(None, settings, defaults, template, imports)
        
    def setitem(self, key, item):
        self.entered_settings[key] = item
        if key in ["batch_size", "shape", "classes"]:
            self.set_variables(None)
            
    @property
    def active_settings(self):
        len_ = self["batch_size"]*self["batches"]        
        act_set = {"len":len_}
        return {**self.settings, **{"len":len_}}
    
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",[self["batch_size"],*self["shape"]],0)
        self.variables.add_variable("y",[self["batch_size"]],0,self["classes"])
        
class RandLongDataset(CodeSection):
    def __init__(self, settings = None):
        template = '''
# Tensor with random values
class Rand:
    def __init__(self):
        self.shape = ${shape}
        self.range = ${range}
        
    def __getitem__(self, i):
        return torch.randint(0,self.range - 1,self.shape), torch.randint(${classes},[1])[0]
        
    def __len__(self):
        return ${len}


trainset = Rand()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=${batch_size}, shuffle=False)

testset = Rand()

testloader = torch.utils.data.DataLoader(testset, batch_size=${batch_size}, shuffle=False)
'''
        defaults = {
            "shape":[200],
            "batches":2,
            "batch_size":4,
            "classes":10,
            "range":10,
        }
        
        imports = set((
            "torch",
        ))
        super().__init__(None, settings, defaults, template, imports)
        
    def setitem(self, key, item):
        self.entered_settings[key] = item
        if key in ["batch_size", "shape", "classes"]:
            self.set_variables(None)
            
    @property
    def active_settings(self):
        len_ = self["batch_size"]*self["batches"]        
        act_set = {"len":len_}
        return {**self.settings, **{"len":len_}}
    
    def set_variables(self, _):
        super().set_variables(None)
        self.variables.add_variable("x",[self["batch_size"],*self["shape"]],0,self["range"])
        self.variables.add_variable("y",[self["batch_size"]],0,self["classes"]) 
        

