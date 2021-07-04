# Neural Question Generation
# Author: Spencer Rose
#
# REFERENCES: 
# Architecture adapted from Alexander Rush, The Annotated Transformer
# http://nlp.seas.harvard.edu/2018/04/03/attention.html
# and
# Joost Bastings. 2018. The Annotated Encoder-Decoder with Attention. 
# https://bastings.github.io/annotated_encoder_decoder/
#
# See also: 
# Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan 
# N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." In 
# Advances in Neural Information Processing Systems, pp. 5998-6008. 2017.

import time
import os
import math
import json
import torch
from torch import nn
from tqdm import tqdm
from config import get_config, print_usage
from model import EncoderDecoder, LossCompute
from enc_dec import Decoder, Encoder, Attention, Generator
from batch import Batch
from tensorboardX import SummaryWriter
from params import Hyperparameters
from utils import print_data_info, print_examples
from preprocess.datawrapper import QASDataset
from preprocess.dataloader import preprocess
from evaluate import eval_model

params = Hyperparameters()

def train(config):
        
    # Load SQuAD data
    data = QASDataset(config)
    print_data_info(config.mode, data)
    
    # Build model from hyperparameters
    model = build_model(data, params, config)
    
    # Check if GPU available
    if params.cuda:
        model.cuda()
        print('CUDA enabled/ Device: '.format(params.device))
    
    # Create Tensorboard summary writer
    tr_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "train"))
    va_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "valid"))
    
    # Create log directory and save directory if it does not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Loss criterion / Optimizer (Adam)
    criterion = nn.NLLLoss(reduction="sum", ignore_index=data.pad_idx)
    optim = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    
    tr_pplx = [] # training perplexity
    va_pplx = [] # validation perplexiy
    
    # initalize epoch offset (override with saved checkpoint)
    epoch_offset = 0
        
    # Prepare checkpoint file and model file to save and load from
    checkpoint_file = os.path.join(config.save_dir, "checkpoint.pth")
    bestmodel_file = os.path.join(config.save_dir, "best_model.pth")
    
    # Check for existing training results. If it existst, and the configuration
    # is set to resume `config.resume==True`, resume from previous training. If
    # not, delete existing checkpoint.
    if os.path.exists(checkpoint_file):
        if config.resume:
            print("Checkpoint found! Resuming")
            # Read checkpoint file.
            load_res = torch.load(checkpoint_file)
            # Resume iterations
            epoch_offset = load_res["epoch"]
            # Resume model
            model.load_state_dict(load_res["model"])
            # Resume optimizer
            optim.load_state_dict(load_res["optimizer"])
        else:
            os.remove(checkpoint_file)
        

    # Iterate over epochs (config default = 8)
    for epoch in range(epoch_offset, config.num_epochs):
        
        # Decay learning rate
        if epoch == params.decay_epoch:
            lr = params.learning_rate * params.learning_rate_decay
            optim = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Training
        print("\nEpoch ", epoch)
        model.train()
        tr_perplexity = run_epoch(
            (Batch(b.src, b.tgt, b.ans, data=data, pad_index=data.pad_idx) 
            for b in data.train_iter),
            model,
            LossCompute(model, criterion, optim)
        )
        tr_pplx.append(tr_perplexity)
        # Write losses to tensorboard 
        tr_writer.add_scalar("perplexity", tr_perplexity, global_step=epoch)
        # Save checkpoint state
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
        }, checkpoint_file)
        # Save current model 
        torch.save({
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
        }, bestmodel_file)
        
        # Validation
        model.eval()
        with torch.no_grad():
            print("Validation step...")
            print_examples(
                (Batch(x.src, x.tgt, x.ans, pad_index=data.pad_idx, validate=True) 
                for x in data.valid_iter),
                model, n=8, 
                src_vocab=data.src.vocab, 
                tgt_vocab=data.tgt.vocab,
                ans_vocab=data.ans.vocab)
            # compute validation loss
            va_perplexity = run_epoch(
                (Batch(b.src, b.tgt, b.ans, pad_index=data.pad_idx, validate=True) 
                for b in data.valid_iter), 
                model, 
                LossCompute(model, criterion, None)
            )
            print("Validation perplexity: %f" % va_perplexity)
            va_pplx.append(va_perplexity)
            # Write losses to tensorboard 
            va_writer.add_scalar("perplexity", va_perplexity, global_step=epoch)

    return tr_pplx, va_pplx

    
def run_epoch(data_iter, model, loss_compute):
    
    """Standard Training and Logging Function"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for i, batch in enumerate(tqdm(data_iter, desc='Progress: '), 1):
        
        out, _, pre_output = model.forward(batch)
        loss = loss_compute(pre_output, batch.tgt_y, batch.nseqs)
        total_loss += loss
        total_tokens += batch.ntokens
        print_tokens += batch.ntokens
        perplexity = math.exp(total_loss / float(total_tokens))

        # Report average batch loss per sequence (at rep_intv intervals)
        if model.training and i % config.rep_intv == 0:
            elapsed = time.time() - start
            print("\nEpoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0
            
    return perplexity



def test(config):
    """Model testing"""

    # Load SQuAD data
    data = QASDataset(config)
    print_data_info(config.mode, data)
    
    # Build model
    model = build_model(data, params, config)

    # Move to GPU if enabled
    if torch.cuda.is_available():
        model = model.cuda()

    # Load saved model and set for testing    
    load_res = torch.load(
            os.path.join(config.save_dir, "best_model.pth"), 
            map_location=lambda storage, loc: storage)
    model.load_state_dict(load_res["model"])
    model.eval()

    # Print some examples
    print_examples(
            (Batch(x.src, x.tgt, x.ans, pad_index=data.pad_idx, validate=True) 
            for x in data.test_iter),
                model, n=5, 
                src_vocab=data.src.vocab, 
                tgt_vocab=data.tgt.vocab,
                ans_vocab=data.ans.vocab)
    # Run evaluation
    results, alphas = eval_model(
            (Batch(x.src, x.tgt, x.ans, pad_index=data.pad_idx) 
            for x in data.test_iter),
            model, 
            src_vocab=data.src.vocab, 
            tgt_vocab=data.tgt.vocab,
            ans_vocab=data.ans.vocab)
    
    # Write results to JSON file
    with open(config.eval_results, 'w+') as f:
        json.dump(results, f)



def build_model(data, params, config):
    '''Build Encoder-Decoder model from hyperparameters'''
    
    # set input size to embedding + 1 for training
    # to handle the answer tagging
    input_size = params.embed_size + 1
        
    return EncoderDecoder(
        Encoder(input_size = input_size, 
                hidden_size = params.hidden_size, 
                num_layers = params.enc_num_layers,
                dropout = params.dropout),
        Decoder(embed_size = params.embed_size, 
                hidden_size = params.hidden_size, 
                attention = Attention(params.hidden_size), 
                num_layers = params.dec_num_layers, 
                dropout = params.dropout),
        nn.Embedding(len(data.src.vocab), params.embed_size),
        nn.Embedding(len(data.tgt.vocab), params.embed_size),
        Generator(params.hidden_size, len(data.tgt.vocab)))
        
        
def main(config):
    
    if config.h:
        print_usage()
    elif config.mode == "preprocess":
        preprocess(config)
    elif config.mode == "train":
        tr_loss, va_loss = train(config)
        print(tr_loss, va_loss)
    elif config.mode == "test":
        test(config)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


if __name__ == "__main__":

    ''' Parse model configuration '''
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)


#
# main.py ends here

