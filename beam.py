#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Helper Functions
# ----------------
# REFERENCES: 
# Wu, Yonghui, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, 
# Wolfgang Macherey, Maxim Krikun et al. "Google's neural machine translation 
# system: Bridging the gap between human and machine translation." arXiv 
# preprint arXiv:1609.08144 (2016).
"""
# Project: UVic CSC586B Final Project: Neural Question Generation
# @author: Spencer Rose, University of Victoria
# Last Updated: April 15, 2019
"""
import numpy as np
import torch
import heapq
import math
from params import Hyperparameters

params = Hyperparameters()

class Beam(object):
# comparison of beams based on score, output_sequence

    def __init__(self, beam_width):
        self.heap = list()
        self.beam_width = beam_width
        self.attn = []
 
    def add(self, score, output, prev_y, attn):
        heapq.heappush(self.heap, (score, output, prev_y, attn))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)
     
    def __iter__(self):
        return iter(self.heap)
    
    
def beam_search(
        model, 
        batch, 
        max_len=100, 
        sos_index=1, 
        eos_index=None, 
        unk_index=0, 
        vocab=None):
    """Beam Search finds candidate tokens for output sequence."""

    with torch.no_grad():
        encoder_hidden, encoder_final = model.encode(
                batch.src, batch.src_mask, batch.src_lengths, batch.ans_mask)
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(batch.src)
        tgt_mask = torch.ones_like(prev_y)
    
    hidden = None
    beam_width = params.beam_width
    prev_beam = Beam(beam_width)
    final_beam = Beam(beam_width)
    # (score, output, previous word index)
    prev_beam.add(0, [], prev_y, [])
    
    # iterate over encoder sequence
    for i in range(max_len):
        curr_beam = Beam(beam_width)
        # iterate over previous beam heap
        for (score, output, prev_y, attn) in prev_beam:
            
            alpha = 0.1
            lp_y = math.pow(5 + len(output), alpha)/math.pow(6, alpha)
            with torch.no_grad():
                # get decoder output
                out, hidden, pre_output = model.decode(
                    encoder_hidden, encoder_final, batch.src_mask, prev_y, tgt_mask, hidden)
                # predict uses log softmax of pre-output layer, which combines
                # decoder state, previous embedding, and context
                # probs shape: [1, src_vocab_dim] 
                probs = model.generator(pre_output[:, -1])
                attn.append(model.decoder.attention.alphas.cpu().numpy())
                
                # get top K candidates
                candidates, c_idx = torch.topk(probs, beam_width)
                c_idx = c_idx.cpu().numpy().T
                candidates = candidates.cpu().numpy()
                
                # iterate over candidates probabilities           
                for (_, next_idx), next_prob in np.ndenumerate(candidates):
                    beam_seq = output[:]
                    prev_y = torch.ones(
                            1, 1).type_as(batch.src).fill_(c_idx[next_idx].item())
                    
                    # filter by max unknowns
                    if beam_seq.count(unk_index) < params.max_num_unks:
                        # calculate coverage penalty
                        cp = 0.5*np.sum(np.log(np.minimum(np.sum(np.array(attn), axis=0), 1.0)))
                        score = (score + next_prob)/lp_y + cp
                        # terminate hypothesis
                        if c_idx[next_idx].item() == eos_index:
                            # penalize end token
                            #coeff = src_len/i
                            coeff = 1
                            final_beam.add(coeff*score, beam_seq, prev_y, attn)
                            
                        else:
                            beam_seq += [c_idx[next_idx].item()]
                            curr_beam.add(score, beam_seq, prev_y, attn)
                    
        prev_beam = curr_beam
        
    #for (score, output, prev_y) in final_beam:
        #print('final beam: score: {}; output: {}; prev_y: {}'.format(score, output, prev_y))
        #print(lookup_words(np.array(output), vocab=vocab))
        
    _, output, _, attn_out = max(final_beam)  
    output = np.array(output)
    
    return output, attn
    
    
# return words from vocabulary indexes
def lookup_words(x, vocab=None):
    """Looks up words in vocabulary dictionary"""
    if vocab is not None:
        x = [vocab.itos[i] for i in x]

    return [str(t) for t in x]