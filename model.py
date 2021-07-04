import torch
from torch import nn
from params import Hyperparameters

# REFERENCES: 
# Architecture adapted from Alexander Rush, The Annotated Transformer
# http://nlp.seas.harvard.edu/2018/04/03/attention.html
# See also: 
# Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan 
# N. Gomez, Łukasz Kaiser, and Illia Polosukhin. "Attention is all you need." In 
# Advances in Neural Information Processing Systems, pp. 5998-6008. 2017.

params = Hyperparameters()

class EncoderDecoder(nn.Module):
    '''
    Sequence-to-Sequence GRU NN
         
    '''
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
        # Move model to GPU if cuda is available
        if torch.cuda.is_available():
            self.cuda()

        
    def forward(self, batch):
        """Take in and process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(
                batch.src, 
                batch.src_mask, 
                batch.src_lengths,
                ans_mask=batch.ans_mask)
        return self.decode(
                encoder_hidden, 
                encoder_final, 
                batch.src_mask, 
                batch.tgt, 
                batch.tgt_mask)
    
    def encode(self, src, src_mask, src_lengths, ans_mask):
        src_embedding = self.src_embed(src)
        # concatenate answer mask
        src_embedding = torch.cat((src_embedding, ans_mask.float()), 2)
        return self.encoder(src_embedding, src_mask, src_lengths)
    
    def decode(self, encoder_hidden, encoder_final, src_mask, tgt, tgt_mask,
               decoder_hidden=None):
        return self.decoder(self.tgt_embed(tgt), encoder_hidden, encoder_final,
                            src_mask, tgt_mask, hidden=decoder_hidden)

    
class LossCompute:
    """A simple loss compute and train function."""
    
    def __init__(self, model, criterion, opt=None):
        self.generator = model.generator
        self.criterion = criterion
        self.opt = opt
    
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm
    
        if self.opt is not None:
            loss.backward()          
            self.opt.step()
            self.opt.zero_grad()
    
        return loss.data.item() * norm
    
    def model_criterion():
        """L2 Regularization"""
    
        def model_loss(model):
            loss = 0
            for name, param in model.named_parameters():
                if "weight" in name:
                    loss += torch.sum(param**2)
    
            return loss * params.l2_reg
    
        return model_loss
 