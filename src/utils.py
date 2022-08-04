from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import copy
from torchsummary import summary
from src.device import device
from torch import Tensor

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def make_tgt_mask(size):
    trg_mask = torch.tril(torch.ones((size, size))).expand(1, size, size)
    return trg_mask.to(device)

def attention(key:Tensor, value:Tensor, query:Tensor, mask=None, dropout: Optional[float] = None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1)) / torch.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
        
    p_attn = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        p_attn = F.dropout(p_attn, dropout)
    
    return torch.matmul(p_attn, value), p_attn
    