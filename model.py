"""
Copyright 2021 Aditya Gomatam.

This file is part of music-transformer (https://github.com/spectraldoy/music-transformer), my project to build and
train a Music Transformer. music-transformer is open-source software licensed under the terms of the GNU General
Public License v3.0. music-transformer is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. music-transformer is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details. A copy of this license can be found within the GitHub repository
for music-transformer, or at https://www.gnu.org/licenses/gpl-3.0.html.
"""

import torch
from math import sqrt
from torch import nn
from hparams import hparams
from layers import EncoderLayer, DecoderLayer, abs_positional_encoding

"""
Implementation of Music Transformer model as an encoder-decoder for converting guitar MIDI to bass MIDI.
Based on Huang et. al, 2018, Vaswani et. al, 2017.
"""

class MusicTransformer(nn.Module):
    """
    Transformer Encoder-Decoder with Relative Attention. Consists of:
        1. Input Embedding for Guitar (Encoder)
        2. Input Embedding for Bass (Decoder)
        3. Absolute Positional Encoding for both
        4. Stack of N EncoderLayers for Guitar
        5. Stack of N DecoderLayers for Bass with Cross-Attention
        6. Final Linear Layer for Bass Prediction
    """
    def __init__(self,
                 d_model=hparams["d_model"],
                 num_layers=hparams["num_layers"],
                 num_heads=hparams["num_heads"],
                 d_ff=hparams["d_ff"],
                 max_rel_dist=hparams["max_rel_dist"],
                 max_abs_position=hparams["max_abs_position"],
                 guitar_vocab_size=hparams["guitar_vocab_size"],  # Vocabulario para guitarra
                 bass_vocab_size=hparams["bass_vocab_size"],      # Vocabulario para bajo
                 bias=hparams["bias"],
                 dropout=hparams["dropout"],
                 layernorm_eps=hparams["layernorm_eps"]):
        """
        Args:
            d_model (int): Tamaño de la dimensión oculta del Transformer.
            num_layers (int): Número de capas en el encoder y decoder.
            num_heads (int): Número de cabezas para la atención multi-cabeza.
            d_ff (int): Dimensión intermedia de las capas FFN.
            max_rel_dist (int): Distancia relativa máxima para incrustaciones posicionales relativas.
            max_abs_position (int): Posición absoluta máxima para codificación posicional sinusoidal.
            guitar_vocab_size (int): Tamaño del vocabulario de guitarra.
            bass_vocab_size (int): Tamaño del vocabulario de bajo.
            bias (bool, optional): Si es False, las capas lineales no aprenderán sesgo. Default: True.
            dropout (float in [0, 1], optional): Tasa de dropout. Default: 0.1.
            layernorm_eps (float, optional): Epsilon para normalización de capas. Default: 1e-6.
        """
        super(MusicTransformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_rel_dist = max_rel_dist
        self.max_position = max_abs_position
        self.guitar_vocab_size = guitar_vocab_size
        self.bass_vocab_size = bass_vocab_size

        # Embedding para la secuencia de guitarra (encoder)
        self.guitar_embedding = nn.Embedding(guitar_vocab_size, d_model)
        # Embedding para la secuencia de bajo (decoder)
        self.bass_embedding = nn.Embedding(bass_vocab_size, d_model)
        # Codificación posicional absoluta para ambas secuencias
        self.positional_encoding = abs_positional_encoding(max_abs_position, d_model)

        # Dropout para entradas
        self.input_dropout = nn.Dropout(dropout)

        # Encoder: pila de capas EncoderLayer
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_rel_dist=max_rel_dist,
                         bias=bias, dropout=dropout, layernorm_eps=layernorm_eps)
            for _ in range(num_layers)
        ])

        # Decoder: pila de capas DecoderLayer con cross-attention
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, max_rel_dist=max_rel_dist,
                         bias=bias, dropout=dropout, layernorm_eps=layernorm_eps)
            for _ in range(num_layers)
        ])

        # Capa final para proyectar al vocabulario de bajo
        self.final = nn.Linear(d_model, bass_vocab_size)

    def forward(self, guitar_seq, bass_seq, guitar_mask=None, bass_mask=None):
        """
        Pase hacia adelante a través del MusicTransformer encoder-decoder.

        Args:
            guitar_seq (torch.Tensor): Secuencia MIDI de guitarra de forma (batch_size, seq_len_guitar).
            bass_seq (torch.Tensor): Secuencia MIDI de bajo parcial de forma (batch_size, seq_len_bass).
            guitar_mask (optional): Máscara para la secuencia de guitarra con 1's en posiciones a enmascarar.
            bass_mask (optional): Máscara para la secuencia de bajo con 1's en posiciones a enmascarar.

        Returns:
            Logits para la predicción del siguiente token de bajo.
        
        """

        # --- Procesamiento de la secuencia de guitarra (encoder) ---
        guitar_emb = self.guitar_embedding(guitar_seq) * sqrt(self.d_model)
        if self.max_position > 0:
            guitar_emb += self.positional_encoding[:, :guitar_emb.shape[1], :]
        guitar_emb = self.input_dropout(guitar_emb)

        memory = guitar_emb
        for layer in self.encoder:
            memory = layer(memory, src_mask=guitar_mask)

        # --- Procesamiento de la secuencia de bajo (decoder) ---
        bass_emb = self.bass_embedding(bass_seq) * sqrt(self.d_model)
        if self.max_position > 0:
            bass_emb += self.positional_encoding[:, :bass_emb.shape[1], :]
        bass_emb = self.input_dropout(bass_emb)

        output = bass_emb
        for layer in self.decoder:
            output = layer(output, memory, tgt_mask=bass_mask, memory_mask=guitar_mask)  # Añadido memory_mask

        return self.final(output)
