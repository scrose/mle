# Helper Functions
# ----------------
# Project: UVic CSC586B Final Project: Neural Question Generation
# Author: Spencer Rose, University of Victoria
# Last Updated: April 13, 2019
#
# REFERENCES: 
# Architecture adapted from Alexander Rush, The Annotated Transformer
# http://nlp.seas.harvard.edu/2018/04/03/attention.html
# and
# Joost Bastings. 2018. The Annotated Encoder-Decoder with Attention. 
# https://bastings.github.io/annotated_encoder_decoder/
#
# See also: 
# [1] Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan 
# N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." In 
# Advances in Neural Information Processing Systems, pp. 5998-6008. 2017.
#
# [2] Wu, Yonghui, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, 
# Wolfgang Macherey, Maxim Krikun et al. "Google's neural machine translation 
# system: Bridging the gap between human and machine translation." arXiv 
# preprint arXiv:1609.08144 (2016).

import numpy as np
import torch
from params import Hyperparameters
from beam import beam_search

params = Hyperparameters()


def greedy_decode(model, batch, max_len=100, sos_index=1, eos_index=None):
    """
    [For Baseline Test]
    Greed Decoder selects most likely word at each step in output sequence.
    Adapted from Joost Bastings. 2018. The Annotated Encoder-Decoder with Attention. 
    https://bastings.github.io/annotated_encoder_decoder/
    """
    # Get encoded output
    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(
                batch.src, batch.src_mask, batch.src_lengths, batch.ans_mask)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(batch.src)
        tgt_mask = torch.ones_like(prev_y)

    output = []
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            out, hidden, pre_output = model.decode(
              encoder_hidden, encoder_final, batch.src_mask,
              prev_y, tgt_mask, hidden)

            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])

        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data.item()
        output.append(next_word)
        prev_y = torch.ones(1, 1).type_as(batch.src).fill_(next_word)
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
    
    output = np.array(output)
        
    # cut off everything starting from </s> 
    # (only when eos_index provided)
    if eos_index is not None:
        first_eos = np.where(output==eos_index)[0]
        if len(first_eos) > 0:
            output = output[:first_eos[0]]      
    
    return output, np.concatenate(attention_scores, axis=1)


def print_data_info(mode, data):
    """ Prints useful information about the data sets. """

    if mode == 'train':
        sdata = data.valid
    elif mode == 'test':
        sdata = data.test
        
    print("{} dataset sizes (number of context/question pairs): {}".format(
            mode.upper(), len(sdata)))
    print()
        
    print("First Example:")
    print("src:", " ".join(vars(sdata[0])['src']))
    print("tgt:", " ".join(vars(sdata[0])['tgt']))
    print("ans:", " ".join(vars(sdata[0])['ans']))
    print()

    print("Most common words (src):")
    print("\n".join(["%10s %10d" % x for x in data.src.vocab.freqs.most_common(10)]), "\n")
    print("Most common words (tgt):")
    print("\n".join(["%10s %10d" % x for x in data.tgt.vocab.freqs.most_common(10)]), "\n")

    print("First 10 words (src):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(data.src.vocab.itos[:10])), "\n")
    print("First 10 words (tgt):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(data.tgt.vocab.itos[:10])), "\n")
    print("Answer Mask (ans):")
    print("\n".join(
        '%02d %s' % (i, t) for i, t in enumerate(data.ans.vocab.itos[:10])), "\n")
    
    print("Number of unique Context tokens: ", len(data.src.vocab))
    print("Number of unique Question tokens: ", len(data.tgt.vocab), "\n")
    
    

def print_examples(example_iter,
                   model, 
                   n=2, 
                   max_len=100, 
                   sos_index=1, 
                   src_eos_index=None, 
                   tgt_eos_index=None, 
                   src_vocab=None, 
                   tgt_vocab=None,
                   ans_vocab=None):
    """
    Prints N examples. Assumes batch size of 1.
    Adapted from Joost Bastings. 2018. The Annotated Encoder-Decoder with Attention. 
    https://bastings.github.io/annotated_encoder_decoder/
    """

    model.eval()
    count = 0
    print()
    
    if src_vocab is not None and tgt_vocab is not None:
        src_eos_index = src_vocab.stoi[params.eos_token]
        tgt_sos_index = tgt_vocab.stoi[params.sos_token]
        tgt_eos_index = tgt_vocab.stoi[params.eos_token]
        tgt_unk_index = tgt_vocab.stoi[params.unk_token]
    else:
        src_eos_index = None
        tgt_sos_index = 1
        tgt_eos_index = None
        tgt_unk_index = 0
        
    for i, batch in enumerate(example_iter):
      
        src = batch.src.cpu().numpy()[0, :]
        tgt = batch.tgt_y.cpu().numpy()[0, :]
        ans = batch.ans.cpu().numpy()[0, :]

        result = None

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        tgt = tgt[:-1] if tgt[-1] == tgt_eos_index else tgt
        ans = ans[:-1] if ans[-1] == src_eos_index else ans
        
        print("\nExample #%d" % (i+1))
        print("source : ", " ".join(lookup_words(src, vocab=src_vocab)))
        print("target : ", " ".join(lookup_words(tgt, vocab=tgt_vocab)))
        print("answer : ", " ".join(lookup_words(ans, vocab=src_vocab)))
        

        # Greedy Decoder
        result, _ = greedy_decode(
              model, batch, max_len=max_len, 
              sos_index=tgt_sos_index, eos_index=tgt_eos_index)
        
        print("Greedy Decode Pred: ", " ".join(
                lookup_words(result, vocab=tgt_vocab)))
            
        # Beam Search
        result, _ = beam_search(
              model, batch, max_len=max_len, 
              sos_index=tgt_sos_index, eos_index=tgt_eos_index, unk_index = tgt_unk_index, vocab=tgt_vocab)
        print("Beam Search Pred: ", " ".join(
                lookup_words(result, vocab=tgt_vocab)))
              
        count += 1
        if count == n:
            break

# return words from vocabulary indexes
def lookup_words(x, vocab=None):
    """Looks up words in vocabulary dictionary"""
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]