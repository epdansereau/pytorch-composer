from pytorch_composer.CodeSection import CodeSection

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

class AG_NEWS(CodeSection):
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
        self.variables.add_variable("y",[self["batch_size"]],0,self["classes"])
        
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

