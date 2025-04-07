import torch
import torch.nn as nn
import math
from typing import Optional,List
from torch import Tensor

class PrepareForMultiHeadAttention(nn.Module):
    """
    <a id="PrepareMHA"></a>

    ## Prepare for multi-head attention

    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """
    def __init__(self, d_model: int, heads:int,d_k:int,bias:bool=True):
        super(PrepareForMultiHeadAttention, self).__init__()
        #length of each input_vec(embedding_seq+positional_encoding)
        self.d_model = d_model
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k
        self.linear = nn.Linear(d_model,heads*d_k,bias=bias)
    def forward(self,x:Tensor):
        '''
        embedding_seq-->q,k,v
        N:SIZE of batch
        S:length of sequence
        D:The dimension of each vector:embedding size
        H:heads
        E:d_k
        Input:[N,S,D]
        Output:[N,S,H,E]
        '''
        head_shape = x.shape[:-1]
        x = self.linear(x)
        #Split the vector into given number of heads
        x = x.view(*head_shape,self.heads,self.d_k)
        return x #(N,S,H,E)


class MultiHeadAttention(nn.Module):
    r"""
    <a id="MHA"></a>

    ## Multi-Head Attention Module

    This computes scaled multi-headed attention for given `query`, `key` and `value` vectors.

    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$

    In simple terms, it finds keys that matches the query, and gets the values of
     those keys.

    It uses dot-product of query and key as the indicator of how matching they are.
    Before taking the $softmax$ the dot-products are scaled by $\frac{1}{\sqrt{d_k}}$.
    This is done to avoid large dot-product values causing softmax to
    give very small gradients when $d_k$ is large.

    Softmax is calculated along the axis of of the sequence (or time).
    """
    def __init__(self, heads:int, d_model: int, dropout: float = 0.1, bias: bool = True):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        # Number of features per head
        assert d_model % heads == 0, "d_model should be divisible by heads"
        self.d_k = d_model // heads
        #q = Wq*x,k,v = Wk*x,v = Wv*x
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        #(N,S,H,E:d_k)
        #dropout
        self.dropout = nn.Dropout(dropout)
        #从seqence维度softmax
        self.softmax = nn.Softmax(dim=1)
        #Scaling factor before softmax
        self.scale = 1 / math.sqrt(self.d_k)
    def get_scores(self,query:Tensor,key:Tensor):
        """QK.T"""#(N,S,H,E)*(N,S,H,E)-->
        return torch.einsum("ishd,jshd->ijbh",query,key)
        
        
        
        
