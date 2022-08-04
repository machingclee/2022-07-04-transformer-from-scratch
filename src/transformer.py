import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import math, copy, time
import matplotlib.pyplot as plt
import seaborn
from torchsummary import summary
from src import utils

seaborn.set_context(context="talk")

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, src_mask, tgt, tgt_mask):
        return self.docode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

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
        
        

    def forward(self, x):
        pass

    