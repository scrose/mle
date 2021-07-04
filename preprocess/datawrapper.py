'''
Wrapper for processed datasets
-----------------------------------
'''
import torchtext as tt
import torch
from params import Hyperparameters

class QASDataset(object):
    """The dataset wrapper for SQuAD dataset. """

    def __init__(self, config):

        # Get hyperparameters
        self.params = Hyperparameters()

        print("Loading SQuAD datasets ...")
        
        # create data fields
        
        # source context
        self.src = tt.data.Field(
            batch_first=True, 
            lower=self.params.lowercase, 
            include_lengths=True,
            unk_token=self.params.unk_token,
            pad_token=self.params.pad_token, 
            init_token=None, 
            eos_token=self.params.eos_token)
        # target question
        self.tgt = tt.data.Field(
            batch_first=True,
            lower=self.params.lowercase, 
            include_lengths=True,
            unk_token=self.params.unk_token, 
            pad_token=self.params.pad_token, 
            init_token=self.params.sos_token, 
            eos_token=self.params.eos_token,
            is_target = True)
        # answer
        self.ans = tt.data.Field(
            batch_first=True,
            include_lengths=True,
            lower=self.params.lowercase, 
            unk_token=self.params.unk_token, 
            pad_token=self.params.pad_token, 
            init_token=None, 
            eos_token=self.params.eos_token,
            is_target = False)
        
        
        # process raw csv data
        self.train = tt.data.TabularDataset(
            path = config.train_data,
            format="csv",
            skip_header=True,
            fields=[('src', self.src), ('tgt', self.tgt), ('ans', self.ans)])
        
        self.valid = tt.data.TabularDataset(
            path = config.valid_data,
            format="csv",
            skip_header=True,
            fields=[('src', self.src), ('tgt', self.tgt), ('ans', self.ans)]) 
        
        self.test = tt.data.TabularDataset(
            path = config.test_data,
            format="csv",
            skip_header=True,
            fields=[('src', self.src), ('tgt', self.tgt), ('ans', self.ans)]) 

        # Get Glove vectors
        # Use downloaded pretrained file if requested
        if config.pretrained:
            weights = tt.vocab.Vectors(config.pretrained)
        else:
            weights = tt.vocab.GloVe(name='840B', dim=300)
        
        # Build vocabularies
        self.src.build_vocab(
                self.train.src, 
                min_freq=self.params.min_freq, 
                vectors=weights)
        self.tgt.build_vocab(
                self.train.tgt, 
                min_freq=self.params.min_freq, 
                vectors=weights)
        self.ans.build_vocab(
                self.train.src, 
                min_freq=self.params.min_freq, 
                vectors=weights)    
        
        # padding index
        self.pad_idx = self.tgt.vocab.stoi[self.params.pad_token]
        
        # initialize word embedding
        init_embed(self.src.vocab)
        init_embed(self.tgt.vocab)
        init_embed(self.ans.vocab)
        
        # Create data iterators for training and validation.
        self.train_iter = tt.data.BucketIterator(
            self.train, 
            batch_size=config.batch_size, 
            train=True,
            sort_within_batch=True,
            sort_key=lambda x: (len(x.src), len(x.tgt)), 
            repeat=False,
            device=self.params.device)

        self.valid_iter = tt.data.Iterator(
            self.valid, 
            batch_size=1,
            train=False, 
            sort=False, 
            repeat=False, 
            device=self.params.device)
        
        self.test_iter = tt.data.Iterator(
            self.test, 
            batch_size=1,
            train=False, 
            sort=False, 
            repeat=False, 
            device=self.params.device)

def init_embed(vocab, init="randn", num_special_toks=2):
    # mode="unk"|"all", all means initialize everything
    emb_vectors = vocab.vectors
    sweep_range = len(vocab)
    running_norm = 0.
    num_non_zero = 0
    total_words = 0
    for i in range(num_special_toks, sweep_range):
        if len(emb_vectors[i, :].nonzero()) == 0:
            # std = 0.5 is based on the norm of average GloVE word vectors
            if init == "randn":
                torch.nn.init.normal_(emb_vectors[i], mean=0, std=0.5)
        else:
            num_non_zero += 1
            running_norm += torch.norm(emb_vectors[i])
        total_words += 1
    print("Average GloVE norm is {}, number of known words are {}, total number of words are {}".format(
        running_norm / num_non_zero, num_non_zero, total_words))
    

def tokenize_mask(data):
    return list(map(int, data.split()))
# datawrapper.py ends here
