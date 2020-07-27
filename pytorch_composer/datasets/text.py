from pytorch_composer.Dataset import Dataset

ROOT = "Path.home() / '.pyt-comp-data'"
NUM_WORKERS = 0


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

class AG_NEWS(Dataset):
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
            
TEXT.build_vocab(train_data${set_vectors})
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
        self.classes = ["World","Sports","Business","Sci/Tech"]
        super().__init__(None, settings, defaults, template, imports)
        
    def set_variables(self, _):
        super().set_variables(None)
        if self["embedding"] is None:
            vocab = {"size":"len(TEXT.vocab)"}
        else:
            vocab = {"embed_dim":get_emb_dim(self["embedding"]), "weights":"TEXT.vocab.vectors"}
        self.variables.add_variable("x",
                                    [self["sequence_length"], self["batch_size"]],
                                    1,
                                    vocab,
                                   )
        self.variables.add_variable("y",[self["batch_size"]],0,self.classes)
        
    @property
    def active_settings(self):
        if self["embedding"] in embeddings:
            set_vectors = ', vectors="{}"'.format(self["embedding"])
        elif self["embedding"] is None:
            set_vectors = ''
        else:
            raise ValueError("Embeddings must be one of: {} or None".format(embeddings))        
        act_set = {"set_vectors":set_vectors}
        return {**self.settings, **act_set}


    
class WikiText2(Dataset):
    def __init__(self, settings = None):
        template = '''
        
root = ${root}
if not os.path.exists(root):
    os.makedirs(root)

TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt${set_vectors})
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


bptt = ${bptt}



class Loader:
    def __init__(self, source):
        self.source = source
        self.nbatch = self.source.size(0) // batch_size
        
    def __iter__(self):
        for i in range(self.nbatch):
            yield self.get_batch(i)
            
    def get_batch(self, i):
        seq_len = min(bptt, len(self.source) - 1 - i)
        data = self.source[i:i+seq_len]
        target = self.source[i+1:i+1+seq_len].view(-1)
        return data, target
    
batch_size = ${batch_size}
eval_batch_size = 10
trainloader = Loader(batchify(train_txt, batch_size))
valloader = Loader(batchify(val_txt, eval_batch_size))
testloader = Loader(batchify(test_txt, eval_batch_size))

'''
        defaults = {
            "batch_size":20,
            "bptt":35,
            "root":ROOT,
            "embedding":"glove.6B.100d",
        }
        
        imports = set((
            ("tqdm", "tqdm"),
            ("pathlib", "Path"),
            "os",
            "torch",
            "torch.nn as nn",
            "torchtext",
            ("torchtext.data.utils", "get_tokenizer")
        ))
        super().__init__(None, settings, defaults, template, imports)
        
    def set_variables(self, _):
        super().set_variables(None)
        if self["embedding"] is None:
            vocab = {"size":"len(TEXT.vocab)"}
        else:
            vocab = {"size":"TEXT.vocab.vectors.shape[0]",
                     "embed_dim":get_emb_dim(self["embedding"]),
                     "weights":"TEXT.vocab.vectors"}
        self.variables.add_variable("x",
                                    [self["bptt"], self["batch_size"]],
                                    1,
                                    vocab,
                                   )
        self.variables.add_variable("y",
                                    [self["bptt"]*self["batch_size"]],
                                    1,
                                    vocab,
                                   )
    @property
    def active_settings(self):
        if self["embedding"] in embeddings:
            set_vectors = ', vectors="{}"'.format(self["embedding"])
        elif self["embedding"] is None:
            set_vectors = ''
        else:
            raise ValueError("Embeddings must be one of: {} or None".format(embeddings))        
        act_set = {"set_vectors":set_vectors}
        return {**self.settings, **act_set}
    
class Multi30k(Dataset):
    #TD: Embeddings
    
    
    def __init__(self, settings = None):
        template = '''
        
SRC = Field(tokenize = "spacy",
            tokenizer_language="de",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            fix_length = ${length})

TRG = Field(tokenize = "spacy",
            tokenizer_language="en",
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            fix_length = ${length})

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))

SRC.build_vocab(train_data, min_freq = ${min_freq})
TRG.build_vocab(train_data, min_freq = ${min_freq})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = ${batch_size}

class IteratorWrapper(BucketIterator):
    def __iter__(self):
        for batch in super().__iter__():
            yield batch.src, batch.trg.view(-1)

trainloader, valid_iterator, testloader = IteratorWrapper.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device)

'''

        defaults = {
            "batch_size":128,
            "length":50,
            "min_freq":2,
        }
        
        imports = set((
            ("tqdm", "tqdm"),
            ("pathlib", "Path"),
            "os",
            "torch",
            ("torchtext.data", "Field, BucketIterator"),
            ("torchtext.datasets", "Multi30k")
        ))
        super().__init__(None, settings, defaults, template, imports)
        
    def set_variables(self, _):
        super().set_variables(None)
        vocab1 = {"size":"len(SRC.vocab)"}
        vocab2 = {"size":"len(TRG.vocab)"}
        self.variables.add_variable("x",
                                    [self["length"], self["batch_size"]],
                                    1,
                                    vocab1,
                                   )
        self.variables.add_variable("y",
                                    [self["length"]*self["batch_size"]],
                                    1,
                                    vocab2,
                                   )






