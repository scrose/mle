#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Helper Functions
# ----------------
# Project: UVic CSC586B Final Project: Neural Question Generation
# Author: Spencer Rose, University of Victoria
# Last Updated: April 13, 2019
#
# REFERENCES: 
# Wu, Yonghui, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, 
# Wolfgang Macherey, Maxim Krikun et al. "Google's neural machine translation 
# system: Bridging the gap between human and machine translation." arXiv 
# preprint arXiv:1609.08144 (2016).
"""
Created on Wed Apr 11 21:25:45 2019

@author: Spencer Rose
"""

from params import Hyperparameters
from nltk import translate as trans
from beam import beam_search
from utils import lookup_words, greedy_decode

params = Hyperparameters()

def eval_model(data_iter,
                   model,  
                   sos_index=1, 
                   src_eos_index=None, 
                   tgt_eos_index=None, 
                   src_vocab=None, 
                   tgt_vocab=None,
                   ans_vocab=None):
    """
    Evaluates Model based on BLEU metric
    """
    
    model.eval()
    
    # evaluation results: predictions
    output = {'ave':'', 'results':{}} 
    output['results'] = []
    
    # store last attention scores
    alphas = []  
    # maximum decoder output length
    max_len = params.dec_max_len
    
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
    
    # Evaluate BLEU and METEOR scores
    for i, batch in enumerate(data_iter):
      
        src = batch.src.cpu().numpy()[0, :]
        tgt = batch.tgt_y.cpu().numpy()[0, :]
        ans = batch.ans.cpu().numpy()[0, :]

        # remove </s> (if it is there)
        src = src[:-1] if src[-1] == src_eos_index else src
        tgt = tgt[:-1] if tgt[-1] == tgt_eos_index else tgt
        ans = ans[:-1] if ans[-1] == src_eos_index else ans
    
        pred = None
        source = lookup_words(src, vocab=src_vocab)
        reference = lookup_words(tgt, vocab=tgt_vocab)
        
        print("src : ", " ".join(source))
        print("tgt : ", " ".join(reference))
      
        # Greedy Decoder
        pred, attention = greedy_decode(
              model, batch, max_len=max_len, 
              sos_index=tgt_sos_index, eos_index=tgt_eos_index)
        
        g_hypothesis = lookup_words(pred, vocab=tgt_vocab)
        g_bleu_score = trans.bleu_score.sentence_bleu([reference], g_hypothesis)
        print("greedy decode : ", " ".join(g_hypothesis))
            
        # Beam Search
        pred, attention = beam_search(
                model, batch, max_len=max_len, 
                sos_index=tgt_sos_index, eos_index=tgt_eos_index, 
                unk_index = tgt_unk_index, vocab=tgt_vocab)
        
        b_hypothesis = lookup_words(pred, vocab=tgt_vocab)
        b_bleu_score = trans.bleu_score.sentence_bleu([reference], b_hypothesis)
        print("beam search : ", " ".join(b_hypothesis))
        
        output['results'].append({
                'tgt':reference, 
                'greedy':g_hypothesis, 
                'beam':b_hypothesis,
                'g_bleu':g_bleu_score,
                'b_bleu':b_bleu_score})
        
        alphas.append(attention)
    
    return output, alphas