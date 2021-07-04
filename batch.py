#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# REFERENCES: 
# Architecture adapted from Alexander Rush, The Annotated Transformer
# http://nlp.seas.harvard.edu/2018/04/03/attention.html
# See also: 
# Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan 
# N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." In 
# Advances in Neural Information Processing Systems, pp. 5998-6008. 2017.
"""
    # Project: UVic CSC586B Final Project: Neural Question Generation
    # @author: Spencer Rose, University of Victoria
    # Last Updated: April 15, 2019
"""
import torch
import numpy as np
import sys
from utils import lookup_words

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """
    def __init__(self, src, tgt, ans, data=None, pad_index=0, validate=False):
        
        src, src_lengths = src
        
        # source context
        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)
        
        # target question
        self.tgt = None
        self.tgt_y = None
        self.tgt_mask = None
        self.tgt_lengths = None
        self.ntokens = None
        self.ans = None
        self.ans_mask = None

        if tgt is not None:
            tgt, tgt_lengths = tgt
            self.tgt = tgt[:, :-1]
            self.tgt_lengths = tgt_lengths
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = (self.tgt_y != pad_index)
            self.ntokens = (self.tgt_y != pad_index).data.sum().item()

        # answer mask
        if ans is not None:
            ans, ans_lengths = ans
            self.ans = ans
            self.ans_lengths = ans_lengths
        
            # create answer tag
            self.ans_mask = make_mask(src, ans, pad_index, validate, data)
            
    
        if torch.cuda.is_available():
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if tgt is not None:
                self.tgt = self.tgt.cuda()
                self.tgt_y = self.tgt_y.cuda()
                self.tgt_mask = self.tgt_mask.cuda()
            
            if ans is not None:
                self.ans = self.ans.cuda()
                self.ans_mask = self.ans_mask.cuda()

# create answer tag mask
def make_mask(src, ans, pad_idx=0, validate=False, data=None):
    # extract answer indexes
    tag_mask = []

    for i, answer in enumerate(ans):
        # initialize tag mask [1, D]
        mask = torch.zeros([src.shape[1], 1], dtype=torch.int32)
        # create answer mask when training
        if not validate:
            # source indexes
            src_ctxt = src[i].cpu().numpy()
            # mask padding indexes
            ans_pad_mask = (answer != pad_idx)
            # answer indexes
            ans_idx = torch.masked_select(answer, ans_pad_mask)[:-1].cpu().numpy()            
            # correlate answer with source indexes
            if len(ans_idx) > 0:
                src_ans_idx = np.where(src_ctxt==ans_idx[0])
                
                # verify correlated indexes
                for idx in src_ans_idx[0]:
                    if np.array_equal(src_ctxt[idx:idx + len(ans_idx)], ans_idx):
                        src_ctxt[idx:idx + len(ans_idx)] = -1
                        mask = (src_ctxt == -1).astype(int)
                        mask = torch.from_numpy(np.expand_dims(mask, axis=1))
                        break
        tag_mask += [mask.long()]
        
    return torch.stack(tag_mask)

    
