import sys
import os

sys.path.append(os.getcwd())

import flax.linen as nn

import pdb
import math
import jax.numpy as jnp
import jax
from typing import Callable

class PositionalEmbedding(nn.Module):
    vocabulary_size : int
    embedding_dim : int
    dropout : float
    
    def setup(self):
        ## We initialize a normal embedding layer.
        self.embedding_layer = nn.Embed(self.vocabulary_size, self.embedding_dim)
        self.dropout_embedding = nn.Dropout(rate=self.dropout)

    def __call__(self, x, *, train): # Is is a tensor of shape: [batch_size, sequence_length].
        assert self.embedding_dim % 2 == 0, "Embedding Dimension should be divisible by two!"

        x = self.embedding_layer(x)
        ## Next, we have to compute a vector of positional embeddings to add.
        pos_embed = jnp.zeros((x.shape[1], self.embedding_dim))
        position = jnp.arange(0, x.shape[1], dtype=jnp.float32)
        div_term = jnp.exp(jnp.arange(0, self.embedding_dim, 2) * (-math.log(10000.0) / self.embedding_dim))
        #pos_embed.at[:, 0::2].set(jnp.sin(jnp.einsum('a,b -> ab', position, div_term)))
        pos_embed = pos_embed.at[:, 0::2].set(jnp.sin(jnp.einsum('a,b -> ab', position, div_term)))
        pos_embed = pos_embed.at[:, 1::2].set(jnp.cos(jnp.einsum('a,b -> ab', position, div_term)))

        ## Finally, we return the dropped out summed result.
        return self.dropout_embedding(x + pos_embed, deterministic=not train)

class EncoderLayer(nn.Module):
    hidden_dim: int
    head_dim : int
    num_heads : int
    dropout : float
    sequence_length : int
    ffn_size : int
    sparsity_parameter: int 
    batch : int
    attn : Callable

    def setup(self):
        self.mha = self.attn

        self.dense_expand = nn.Dense(self.ffn_size)
        self.dense_contract = nn.Dense(self.hidden_dim)

        ## We need two layer_norm layers.
        self.layer_norm_one = nn.LayerNorm()
        self.layer_norm_two = nn.LayerNorm()

        ## We need three dropout layers.
        self.dropout_one = nn.Dropout(self.dropout)

    def __call__(self, x, mask, step, *, train):

        ## we first compute the attention value.
        attn = self.mha(hidden_states=x, attention_mask=mask)

        ## Then we have to compute the layer norm of the addition.
        attn_prev_ffn = self.layer_norm_one(attn + x)

        ## Then we have to put it through a dense ffn Layer.
        attn = self.dense_expand(attn_prev_ffn)
        attn = self.dense_contract(attn)
        attn = self.dropout_one(attn, deterministic=not train)

        ## Then we have to put it through the last layer-norm.
        attn = self.layer_norm_two(attn + attn_prev_ffn)

        ## Then this is our final value.
        return attn

class Encoder(nn.Module):
    hidden_dim: int
    head_dim : int
    num_heads : int
    dropout : float
    sequence_length : int
    ffn_size : int
    encoder_layers : int
    batch : int
    sparsity_parameter : int
    attn : Callable

    def setup(self):
        self.encoders = [EncoderLayer(self.hidden_dim, self.head_dim, 
                                        self.num_heads, self.dropout, 
                                        self.sequence_length, self.ffn_size, 
                                        batch=self.batch, sparsity_parameter=self.sparsity_parameter, attn=self.attn) for _ in range(self.encoder_layers)]

    def __call__(self, x, mask, step, *, train):
        for enc in self.encoders:
            x = enc(x, mask=mask, step=step, train=train)

        return x

## We make this an encoder-only model for performance benchmarking. ##
class Transformer(nn.Module):
    hidden_dim: int
    head_dim : int
    num_heads : int
    dropout : float
    sequence_length : int
    ffn_size : int
    num_layers : int
    vocabulary_size : int
    encoder_only : bool
    batch : int
    sparsity_parameter : int
    attn : Callable

    def setup(self):
        self.positional_embedding = PositionalEmbedding(self.vocabulary_size, self.hidden_dim, self.dropout)

        self.encoder = Encoder(self.hidden_dim, self.head_dim, self.num_heads, 
                               self.dropout, self.sequence_length, self.ffn_size, 
                               self.num_layers, batch=self.batch, sparsity_parameter=self.sparsity_parameter, attn=self.attn)

        self.last_ffn = nn.Dense(self.vocabulary_size)

    def __call__(self, encoder_input, mask, *, train):
            ## Over here, x is one input.
        encoder_input = self.positional_embedding(encoder_input, train=train)
        encoder_output = self.encoder(encoder_input, mask, False, train=train)
        return encoder_output 
