#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------
# Encoder-Decoder
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
# Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan 
# N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." In 
# Advances in Neural Information Processing Systems, pp. 5998-6008. 2017.

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class Encoder(nn.Module):
    '''
    Seq-to-Seq GRU Encoder 
    Adapted from Joost Bastings. 2018. The Annotated Encoder-Decoder with Attention. 
    https://bastings.github.io/annotated_encoder_decoder/
    '''
    
    def __init__(
            self, 
            input_size, 
            hidden_size, 
            num_layers=1, 
            dropout=0.0
        ):
        
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        
        # define RNN unit: GRU
        self.gru = nn.GRU(
                input_size, 
                hidden_size, 
                num_layers,
                batch_first=True, 
                bidirectional=True, 
                dropout=dropout
        )
        
    def forward(self, data, mask, seq_lens):
        """
        Applies a bidirectional RNN to sequence of embeddings x.
        [data] requires dimensions [Batch, Step, Dim].
        """
        packed_data = pack_padded_sequence(data, seq_lens, batch_first=True)
        output, h_n = self.gru(packed_data)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # intermediate hidden states
        fw_h = h_n[0:h_n.size(0):2]
        bk_h = h_n[1:h_n.size(0):2]
        
        # [num_layers, batch, 2*dim]
        h_n = torch.cat([fw_h, bk_h], dim=2)

        return output, h_n


class Decoder(nn.Module):
    '''
    Seq2Seq Decoder 
    Adapted from Joost Bastings. 2018. The Annotated Encoder-Decoder with Attention. 
    https://bastings.github.io/annotated_encoder_decoder/
    '''
    def __init__(
            self, 
            embed_size, 
            hidden_size, 
            attention, 
            num_layers=1, 
            dropout=0.5,
            bridge=True
        ):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
                 
        self.gru = nn.GRU(
                embed_size + 2*hidden_size, 
                hidden_size, 
                num_layers,
                batch_first=True, 
                dropout=dropout
        )
                 
        # to initialize from the final encoder state
        self.bridge = nn.Linear(
                2*hidden_size,
                hidden_size,
                bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(
                hidden_size + 2*hidden_size + embed_size,
                hidden_size,
                bias=False
        )
        
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, 
            proj_key=proj_key,
            value=encoder_hidden, 
            mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.gru(rnn_input, hidden)
        
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output
    
    def forward(self, trg_embed, encoder_hidden, encoder_final, 
                src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""
                                         
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
        
        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)
        
        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
              prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final)) 


'''
Attention Class 
--------------------------
Implements Bahdanau (MLP) attention

parameters: 
    - hidden_size: 
    - key_size: 
    - query_size: 
    - params: set of basic hyperparameters
returns: 
    - Bahdanau attention module
'''
class Attention(nn.Module):
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(Attention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attentional scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computed.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        # Applies batch matrix-matrix product of matrices
        context = torch.bmm(alphas, value)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas


class LuongAttention(nn.Module):
    """
    LuongAttention from Effective Approaches to Attention-based Neural Machine Translation
    https://arxiv.org/pdf/1508.04025.pdf
    """

    def __init__(self, dim):
        super(LuongAttention, self).__init__()
        self.W = nn.Linear(dim, dim, bias=False)

    def score(self, decoder_hidden, encoder_out):
        # linear transform encoder out (seq, batch, dim)
        encoder_out = self.W(encoder_out)
        # (batch, seq, dim) | (2, 15, 50)
        encoder_out = encoder_out.permute(1, 0, 2)
        # (2, 15, 50) @ (2, 50, 1)
        return encoder_out @ decoder_hidden.permute(1, 2, 0)

    def forward(self, decoder_hidden, encoder_out):
        energies = self.score(decoder_hidden, encoder_out)
        mask = F.softmax(energies, dim=1)  # batch, seq, 1
        context = encoder_out.permute(
            1, 2, 0) @ mask  # (2, 50, 15) @ (2, 15, 1)
        context = context.permute(2, 0, 1)  # (seq, batch, dim)
        mask = mask.permute(2, 0, 1)  # (seq2, batch, seq1)
        return context, mask


'''
Generator Class
---------------------------
projects the pre-output layer (input sequence in the forward function) 
to obtain the output layer, so that the final dimension is the 
target vocabulary size.
'''
class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
    
    