import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
import seaborn
from torchsummary import summary
from src import utils
from torch import Tensor
from typing import Optional
from device import device
seaborn.set_context(context="talk")

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, src_mask, tgt, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Encoder(nn.Module):
    def __init__(self, encoder_layer, N):
        super(Encoder, self).__init__()
        self.encoder_layers = utils.clones(encoder_layer, N)
 
    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x
    
class AddAndNorm(nn.Module):
    def __init__(self, embed_size, dropout):
        super(AddAndNorm, self).__init__()
        self.embed_size = embed_size
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x, conn_tgt):
        return self.dropout(self.norm(x + conn_tgt))

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.embed_size = embed_size
        self.add_and_norms = utils.clones(AddAndNorm(embed_size, dropout), 2)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = dropout

    def forward(self, x, mask):
        x = self.add_and_norms[0](x, self.self_attn(x, x, x, mask))
        x = self.add_and_norms[1](x, self.feed_forward(x))
        return x

        
class Decoder(nn.Module):
    def __init__(self, decoder_layer, N):
        super(Decoder, self).__init__()
        self.decoder_layers = utils.clones(decoder_layer, N)
        self.norm = nn.LayerNorm(decoder_layer.embed_size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, embed_size, self_attn, src_attn, feed_forward, dropout):
        self.embed_size = embed_size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout
        self.add_and_norms = utils.clones(AddAndNorm(embed_size, dropout), 3)
        super(DecoderLayer, self).__init__()

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.add_and_norms[0](x, self.self_attn(x, x, x, tgt_mask))
        x = self.add_and_norms[1](x, self.src_attn(m, m, x, src_mask))
        
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_head = d_model // heads
        self.heads = heads
        self.linears = utils.clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
       
        
        

    def forward(self, key:Tensor, value:Tensor, query:Tensor, mask:Optional[Tensor]=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
            
        n_batches = query.size(0)
        
        key, value, query = [linear(x).reshape(n_batches, -1, self.heads, self.d_head).transpose(1,2)
                             for linear, x in (self.linears, (key, value, query))]
        
        x, self.attn = utils.attention(key, value, query, mask=mask, dropout=self.dropout)
        x = x.transpose(1,2).continguous().view(n_batches, -1, self.heads*self.d_head)
        return self.linear[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        self.w_2(self.dropout(F.relu(self.w_1(x)))) 
        
class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__() 
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeors(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)/d_model)*(-math.log(10000.0))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, src_pad_idx=0,  trg_pad_idx=0):
        super(Transformer, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        self.encoder_decoder = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embeddings(d_model,src_vocab), c(position)),
            nn.Sequential(Embeddings(d_model,tgt_vocab), c(position))
        )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        for p in self.encoder_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(device)
    
    def make_tgt_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(device)      

    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        return self.encoder_decoder(src, src_mask, tgt, tgt_mask)
    