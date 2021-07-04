#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Reference:
# Zhou, Qingyu, Nan Yang, Furu Wei, Chuanqi Tan, Hangbo Bao, and Ming Zhou. 
# "Neural question generation from text: A preliminary study." In National 
# CCF Conference on Natural Language Processing and Chinese Computing, 
# pp. 662-671. Springer, Cham, 2017.
"""
    # Project: UVic CSC586B Final Project: Neural Question Generation
    # @author: Spencer Rose, University of Victoria
    # Last Updated: April 15, 2019
"""
import torch

class Hyperparameters(object):
    '''A set of basic hyperparameters'''

    def __init__(self):
        # CUDA settings
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0') if self.cuda else torch.device('cpu')
        
        # special parameters
        self.unk_token = "<unk>"
        self.pad_token = "<pad>" 
        self.sos_token = "<s>"
        self.eos_token = "</s>"
        self.lowercase = True
        self.min_freq = 1
        
        # answer tagging
        # tag B denotes the start ofan answer
        # tag I continues the answer
        # tag O marks words that do not form part of an answer
        self.b_token = "<b>"
        self.i_token = "<i>"
        self.o_token = "<o>"
        
        # NN parameters
        self.embed_size = 300
        self.embed_trainable = False
        self.hidden_size = 512
        self.enc_num_layers = 2
        self.dec_num_layers = 2
        self.dropout = 0.5
        self.output_decoder = 'beam' # "beam'|'greedy'
        self.beam_width = 3
        self.dec_max_len = 30
        self.max_num_unks = 3
    
        # Learning parameters
        self.batch_size = 64
        self.learning_rate = 0.0001
        self.learning_rate_decay = 0.5
        self.l2_reg = 1e-6
        self.decay_epoch = 8
