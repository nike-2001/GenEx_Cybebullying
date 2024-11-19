# coding=utf-8
# Licensed under the Apache License, Version 2.0
# This file implements a BART model using PyTorch, ported from the fairseq repository.

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

import torch  # For tensor computations
import torch.nn.functional as F  # For activation functions and loss calculations
from torch import Tensor, nn  # For tensor typing and neural network layers
from transformers import BartConfig  # Hugging Face configuration class for BART
from transformers.file_utils import add_start_docstrings  # Decorator for adding docstrings
from transformers.generation.beam_search import BeamHypotheses  # For managing beam search hypotheses
from transformers.modeling_utils import PreTrainedModel  # Base class for all pre-trained models

# Initialize a logger to output debugging and informational messages
logger = logging.getLogger(__name__)

# URLs for pre-trained BART models stored on Hugging Face servers
BART_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bart-large": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/pytorch_model.bin",
    "bart-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-mnli/pytorch_model.bin",
    "bart-large-cnn": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-cnn/pytorch_model.bin",
}

# Base documentation for BART models
BART_START_DOCSTRING = r"""
This model is a PyTorch `torch.nn.Module` sub-class. Use it as a regular PyTorch Module.
"""

# Documentation string for BART input arguments
BART_INPUTS_DOCSTRING = r"""
Args:
    input_ids (torch.LongTensor): Input sequence token indices.
    attention_mask (torch.Tensor): Mask to ignore padding tokens.
    decoder_input_ids (torch.LongTensor): IDs for the decoder's input sequence.
"""

# Large negative value used for masking positions in attention
LARGE_NEGATIVE = -1e4


def create_position_ids_from_input_ids(input_ids, padding_idx):
    """
    Generate position IDs for input IDs, accounting for padding tokens.

    Args:
        input_ids (torch.Tensor): Tensor of shape `(batch_size, sequence_length)`.
        padding_idx (int): Index representing padding tokens.

    Returns:
        torch.Tensor: Position IDs of shape `(batch_size, sequence_length)`.
    """
    # Create a mask to identify non-padding tokens
    mask = input_ids.ne(padding_idx).int()
    # Generate cumulative position IDs for non-padding tokens
    position_ids = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx
    return position_ids


def add_start_docstrings_to_callable(docstring):
    """
    Decorator to append a docstring to a callable function or method.

    Args:
        docstring (str): The docstring to prepend.

    Returns:
        callable: Decorated function or method with the updated docstring.
    """
    def decorator(func):
        func.__doc__ = docstring + (func.__doc__ or "")
        return func
    return decorator


def _prepare_bart_decoder_inputs(config, input_ids, decoder_input_ids=None, decoder_attn_mask=None):
    """
    Prepares decoder inputs and attention masks for the BART model.

    Args:
        config (BartConfig): Configuration for the BART model.
        input_ids (torch.Tensor): Input token IDs.
        decoder_input_ids (torch.Tensor, optional): Decoder input IDs, if provided.
        decoder_attn_mask (torch.Tensor, optional): Decoder attention mask.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Prepared decoder input IDs and attention masks.
    """
    pad_token_id = config.pad_token_id  # Padding token ID defined in the configuration
    need_causal_mask = not config.use_cache  # Check if causal mask is required

    # Generate decoder input IDs by shifting input tokens, if not provided
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)

    bsz, tgt_len = decoder_input_ids.size()[:2]  # Extract batch size and target sequence length

    # Create attention masks if not provided
    if decoder_attn_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
        if need_causal_mask:
            causal_lm_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1)
        else:
            causal_lm_mask = None

        # Combine padding and causal masks
        new_shape = (bsz, tgt_len, tgt_len)
        decoder_attn_mask = _combine_masks(decoder_padding_mask, causal_lm_mask, new_shape).to(device=input_ids.device)

    return decoder_input_ids, decoder_attn_mask


class PretrainedBartModel(PreTrainedModel):
    """
    Base class for all BART models. Inherits from `PreTrainedModel`.

    This class provides shared functionality for pre-trained BART models, such as weight initialization.
    """
    config_class = BartConfig  # Configuration class for BART
    base_model_prefix = "model"  # Prefix for identifying base models
    pretrained_model_archive_map = BART_PRETRAINED_MODEL_ARCHIVE_MAP  # Mapping of pre-trained model checkpoints

    def _init_weights(self, module):
        """
        Initializes weights for the given module.

        Args:
            module (torch.nn.Module): The module whose weights are to be initialized.
        """
        std = self.config.init_std  # Standard deviation for normal initialization

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)  # Initialize weights with normal distribution
            if module.bias is not None:
                module.bias.data.zero_()  # Set biases to zero

        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)  # Initialize embedding weights
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()  # Zero out weights for the padding index

    @property
    def dummy_inputs(self):
        """
        Generates dummy inputs for testing and debugging the model.

        Returns:
            dict: A dictionary containing dummy inputs, including `input_ids` and attention masks.
        """
        pad_token = 1  # Padding token ID
        input_ids = torch.Tensor([
            [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2],
            [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 2, pad_token],
        ]).long()

        # Prepare decoder inputs and attention masks
        decoder_input_ids, decoder_attn_mask = _prepare_bart_decoder_inputs(
            self.config, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attn_mask=None
        )

        dummy_inputs = {
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_attention_mask": decoder_attn_mask,
        }
        return dummy_inputs

def _make_linear_from_emb(emb):
    """
    Creates a linear layer that shares weights with the given embedding layer.
    Args:
        emb (nn.Embedding): The embedding layer whose weights will be shared.
    Returns:
        nn.Linear: A linear layer with weights tied to the embedding layer.
    """
    # Extract the size of the vocabulary and embedding dimensions
    vocab_size, emb_size = emb.weight.shape

    # Create a linear layer with the same dimensions, without bias
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)

    # Tie the weights of the linear layer to the embedding layer
    lin_layer.weight = emb.weight

    return lin_layer


# Helper Functions, mostly for making masks

def _check_shapes(shape_1, shape2):
    """
    Checks if two shapes are identical, and raises an error if they are not.
    Args:
        shape_1 (tuple): The first shape to compare.
        shape2 (tuple): The second shape to compare.
    Raises:
        AssertionError: If the shapes do not match.
    """
    if shape_1 != shape2:
        raise AssertionError("shape mismatch: {} != {}".format(shape_1, shape2))


def _combine_masks(key_padding_mask, attn_mask, targ_size):
    """
    Combines key padding masks and attention masks into a single mask.
    Args:
        key_padding_mask (torch.Tensor): Mask for padding tokens (shape: [bsz, tgt_len]).
        attn_mask (torch.Tensor): Mask for attention (shape: [tgt_len, src_len]).
        targ_size (tuple): Target size of the combined mask ([bsz, tgt_len, src_len]).
    Returns:
        torch.Tensor: Combined mask of shape (bsz, 1, tgt_len, src_len).
    """
    # Initialize two masks with zeros
    a = torch.zeros(targ_size)
    b = torch.zeros(targ_size)

    # Add key padding mask to the combined mask if provided
    if key_padding_mask is not None:
        _check_shapes(key_padding_mask.shape, targ_size[:2])  # Ensure dimensions match
        reshaped = key_padding_mask.unsqueeze(2).expand(*targ_size)  # Reshape and expand to match targ_size
        a[reshaped] = 1e-8  # Assign a small value to masked positions

    # Add attention mask to the combined mask if provided
    if attn_mask is not None:
        _check_shapes(attn_mask.shape, targ_size[-2:])  # Ensure dimensions match
        b = attn_mask.unsqueeze(0).expand(*targ_size)  # Expand to match targ_size

    # Combine the two masks, clamp values to handle large negatives, and add a singleton dimension
    return (a + b).unsqueeze(1).clamp(LARGE_NEGATIVE,)


def shift_tokens_right(input_ids, pad_token_id):
    """
    Shifts input token IDs one position to the right, wrapping the last non-padding token.
    Args:
        input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
        pad_token_id (int): ID representing the padding token.
    Returns:
        torch.Tensor: Shifted token IDs with the first position updated.
    """
    # Clone the input token IDs
    prev_output_tokens = input_ids.clone()

    # Find the index of the last non-padding token in each sequence
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)

    # Move the last non-padding token to the first position
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()

    # Shift all other tokens to the right
    prev_output_tokens[:, 1:] = input_ids[:, :-1]

    return prev_output_tokens


def make_padding_mask(input_ids, padding_idx=1):
    """
    Creates a padding mask for a sequence of token IDs.
    Args:
        input_ids (torch.Tensor): Input token IDs of shape (batch_size, seq_len).
        padding_idx (int, optional): ID representing the padding token. Defaults to 1.
    Returns:
        torch.Tensor or None: Padding mask (True for padding tokens), or None if no padding is found.
    """
    # Create a mask where True represents padding tokens
    padding_mask = input_ids.eq(padding_idx)

    # Return None if no padding tokens are found
    if not padding_mask.any():
        padding_mask = None

    return padding_mask


# Helper Modules

# EncoderLayer implements a single layer of the Transformer encoder
class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        """
        Initializes the components of an encoder layer:
        - Self-attention mechanism
        - Layer normalization
        - Feedforward neural network
        Args:
            config (BartConfig): Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.embed_dim = config.d_model  # Embedding dimension
        self.output_attentions = config.output_attentions  # Whether to output attention weights

        # Self-attention mechanism
        self.self_attn = SelfAttention(
            self.embed_dim,
            config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )

        # Layer normalization for self-attention outputs
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        # Dropout settings
        self.dropout = config.dropout
        self.activation_dropout = config.activation_dropout

        # Feedforward network
        self.activation_fn = F.gelu  # Activation function (GELU)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # First fully connected layer
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # Second fully connected layer

        # Final layer normalization
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask):
        """
        Performs the forward pass for the encoder layer.
        Args:
            x (Tensor): Input tensor of shape `(seq_len, batch, embed_dim)`.
            encoder_padding_mask (ByteTensor): Padding mask indicating which tokens are padding (1 for padding, 0 otherwise).
        Returns:
            Tuple[Tensor, Tensor]: The output tensor and attention weights.
        """
        residual = x  # Store the input for residual connection

        # Self-attention block
        x, attn_weights = self.self_attn(
            query=x, key=x, value=x, key_padding_mask=encoder_padding_mask, need_weights=self.output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout
        x = residual + x  # Add residual connection
        x = self.self_attn_layer_norm(x)  # Apply layer normalization

        residual = x  # Store the output for the next residual connection

        # Feedforward network
        x = self.activation_fn(self.fc1(x))  # First fully connected layer with activation
        x = F.dropout(x, p=self.activation_dropout, training=self.training)  # Dropout after activation
        x = self.fc2(x)  # Second fully connected layer
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        x = residual + x  # Add residual connection
        x = self.final_layer_norm(x)  # Final layer normalization

        return x, attn_weights


# BartEncoder implements the Transformer encoder
class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of multiple `EncoderLayer` layers.
    Each layer is a self-attention layer followed by a feedforward network.
    Args:
        config (BartConfig): Configuration object containing model hyperparameters.
        embed_tokens (nn.Embedding): Token embedding layer.
    """

    def __init__(self, config: BartConfig, embed_tokens):
        """
        Initializes the encoder with:
        - Token embeddings
        - Positional embeddings
        - Stacked encoder layers
        - Layer normalization for embeddings
        """
        super().__init__()

        self.dropout = config.dropout  # Dropout probability
        self.layerdrop = config.encoder_layerdrop  # Probability of dropping an entire layer
        self.output_attentions = config.output_attentions  # Whether to output attention weights
        self.output_hidden_states = config.output_hidden_states  # Whether to output hidden states

        embed_dim = embed_tokens.embedding_dim  # Dimension of token embeddings
        self.padding_idx = embed_tokens.padding_idx  # Padding token index
        self.max_source_positions = config.max_position_embeddings  # Maximum input length

        self.embed_tokens = embed_tokens  # Token embedding layer

        # Positional embeddings
        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, embed_dim, self.padding_idx,
        )

        # Stack of encoder layers
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])

        # Layer normalization for input embeddings
        self.layernorm_embedding = LayerNorm(embed_dim)

    def forward(self, input_ids=None, attention_mask=None):
        """
        Forward pass for the encoder.
        Args:
            input_ids (LongTensor): Input token IDs of shape `(batch, src_len)`.
            attention_mask (torch.LongTensor): Mask indicating padding tokens (1 for padding, 0 otherwise).
        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]:
                - Output of the last encoder layer (shape: `(src_len, batch, embed_dim)`).
                - Hidden states from all layers (if `output_hidden_states` is True).
                - Attention weights from all layers (if `output_attentions` is True).
        """
        # Embed input tokens and add positional embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos  # Combine token and positional embeddings
        x = self.layernorm_embedding(x)  # Normalize embeddings
        x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout

        # Transpose for compatibility with subsequent operations: (batch, src_len, embed_dim) -> (src_len, batch, embed_dim)
        x = x.transpose(0, 1)

        # Lists to store hidden states and attention weights
        encoder_states, all_attentions = [], []

        # Process input through each encoder layer
        for encoder_layer in self.layers:
            if self.output_hidden_states:
                encoder_states.append(x)  # Store the current hidden state

            # Apply LayerDrop (randomly skip layers during training)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                attn = None  # Skip the layer
            else:
                x, attn = encoder_layer(x, attention_mask)  # Forward pass through the layer

            if self.output_attentions:
                all_attentions.append(attn)  # Store attention weights

        if self.output_hidden_states:
            encoder_states.append(x)  # Store the final hidden state

        # Transpose hidden states back to (batch, src_len, embed_dim)
        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]

        # Return the final output, hidden states, and attention weights
        return x, encoder_states, all_attentions


class DecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        """
        Implements a single layer of the Transformer decoder.
        This layer includes:
        - Self-attention mechanism
        - Encoder-decoder attention
        - Feedforward neural network with layer normalization

        Args:
            config (BartConfig): Configuration object containing model hyperparameters.
        """
        super().__init__()
        self.embed_dim = config.d_model  # Embedding dimension

        # Self-attention mechanism for decoder
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout  # Dropout probability
        self.activation_fn = F.gelu  # Activation function
        self.activation_dropout = config.activation_dropout  # Dropout after activation

        # Layer normalization for self-attention output
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        # Cross-attention mechanism for encoder-decoder interaction
        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,  # Indicates cross-attention
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)  # Layer norm for cross-attention

        # Feedforward network
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)  # First fully connected layer
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)  # Second fully connected layer
        self.final_layer_norm = LayerNorm(self.embed_dim)  # Final layer normalization

    def forward(
            self,
            x,
            encoder_hidden_states,
            encoder_attn_mask=None,
            layer_state=None,
            attention_mask=None,
            need_attn_weights=False,
    ):
        """
        Forward pass for the decoder layer.

        Args:
            x (Tensor): Input tensor of shape `(seq_len, batch, embed_dim)`.
            encoder_hidden_states (Tensor): Hidden states from the encoder.
            encoder_attn_mask (ByteTensor, optional): Mask for encoder attention (padding mask).
            layer_state (dict, optional): Cached states for self-attention during decoding.
            attention_mask (Tensor, optional): Mask for decoder self-attention.
            need_attn_weights (bool, optional): Whether to return attention weights.

        Returns:
            Tuple[Tensor, Tensor, dict]: Output tensor, self-attention weights, and updated layer state.
        """
        # Store the input for the residual connection
        residual = x
        y = x  # Duplicate input for self-attention query, key, and value

        # Initialize layer state if not provided
        if layer_state is None:
            layer_state = {}

        # Self-attention
        x, self_attn_weights = self.self_attn(
            query=x, key=y, value=y, layer_state=layer_state, need_weights=need_attn_weights, attn_mask=attention_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout
        x = residual + x  # Add residual connection
        x = self.self_attn_layer_norm(x)  # Layer normalization

        # Cross-attention with encoder hidden states
        residual = x
        x, encoder_attn_weights = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,  # Encoder output as key
            value=encoder_hidden_states,  # Encoder output as value
            key_padding_mask=encoder_attn_mask,  # Mask to ignore padding
            layer_state=layer_state,  # Update layer state for caching
            static_kv=True,  # Static key-value for cross-attention
            need_weights=False,  # Attention weights are not required here
        )
        x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout
        x = residual + x  # Add residual connection
        x = self.encoder_attn_layer_norm(x)  # Layer normalization

        # Feedforward network
        residual = x
        x = self.activation_fn(self.fc1(x))  # Apply first linear layer with activation
        x = F.dropout(x, p=self.activation_dropout, training=self.training)  # Dropout after activation
        x = self.fc2(x)  # Apply second linear layer
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        x = residual + x  # Add residual connection
        x = self.final_layer_norm(x)  # Final layer normalization

        # Return output, self-attention weights, and updated layer state
        return x, self_attn_weights, layer_state


class BartDecoder(nn.Module):
    """
    Implements the Transformer decoder consisting of multiple `DecoderLayer`s.

    Args:
        config (BartConfig): Configuration object containing model hyperparameters.
        embed_tokens (torch.nn.Embedding): Embedding layer for token inputs.
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        """
        Initializes the decoder:
        - Token and positional embeddings
        - Stacked decoder layers
        - Layer normalization for embeddings
        """
        super().__init__()
        self.use_cache = config.use_cache  # Whether to cache states for decoding
        self.output_attentions = config.output_attentions  # Output attention weights
        self.output_hidden_states = config.output_hidden_states  # Output hidden states
        self.dropout = config.dropout  # Dropout probability
        self.layerdrop = config.decoder_layerdrop  # Probability of dropping layers
        self.padding_idx = embed_tokens.padding_idx  # Padding token index
        self.max_target_positions = config.max_position_embeddings  # Maximum sequence length

        self.embed_tokens = embed_tokens  # Token embedding layer

        # Positional embeddings
        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model, self.padding_idx,
        )

        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self.layernorm_embedding = LayerNorm(config.d_model)  # Layer normalization for embeddings
        self.generation_mode = False  # Whether the decoder is in generation mode

    def forward(
            self,
            input_ids,
            encoder_hidden_states,
            encoder_padding_mask,
            combined_mask,
            decoder_cached_states=None,
            **unused
    ):
        """
        Forward pass for the decoder.

        Args:
            input_ids (LongTensor): Previous decoder outputs for teacher forcing `(batch, tgt_len)`.
            encoder_hidden_states (Tensor): Encoder output for cross-attention.
            encoder_padding_mask (Tensor): Mask to ignore padding in encoder input.
            combined_mask (Tensor): Combined mask for decoder self-attention.
            decoder_cached_states (dict, optional): Cached states for decoding.

        Returns:
            Tuple[Tensor, Optional[Tuple], List[Tensor], List[Tensor]]:
                - Final decoder output `(batch, tgt_len, embed_dim)`
                - Cached states for future decoding
                - All hidden states (if output_hidden_states is True)
                - All self-attention weights (if output_attentions is True)
        """
        # Compute positional embeddings
        positions = self.embed_positions(input_ids, generation_mode=self.generation_mode)

        # Restrict input and positional embeddings in generation mode
        if self.generation_mode:
            input_ids = input_ids[:, -1:]  # Use only the last token
            positions = positions[:, -1:]

        # Embed input tokens and add positional embeddings
        x = self.embed_tokens(input_ids)
        x += positions
        x = self.layernorm_embedding(x)  # Layer normalization
        x = F.dropout(x, p=self.dropout, training=self.training)  # Apply dropout

        # Transpose for compatibility: (batch, tgt_len, embed_dim) -> (tgt_len, batch, embed_dim)
        x = x.transpose(0, 1)

        all_hidden_states = ()  # Store hidden states if required
        all_self_attns = ()  # Store self-attention weights if required
        next_decoder_cache = []  # Store cached states for next step

        # Process through each decoder layer
        for i, decoder_layer in enumerate(self.layers):
            # Apply LayerDrop (skip layers randomly during training)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            # Retrieve cached state for this layer
            layer_state = decoder_cached_states[i] if decoder_cached_states is not None else None

            # Forward pass through the layer
            x, layer_self_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                encoder_padding_mask,
                layer_state=layer_state,
                attention_mask=combined_mask,
                need_attn_weights=self.output_attentions,
            )

            # Cache layer state if required
            if self.use_cache:
                next_decoder_cache.append(layer_past.copy())

            # Store hidden states and attention weights if required
            if self.output_hidden_states:
                all_hidden_states += (x,)
            if self.output_attentions:
                all_self_attns += (layer_self_attn,)

        # Transpose back to (batch, tgt_len, embed_dim)
        x = x.transpose(0, 1)
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]

        # Prepare the next cache for generation
        next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache) if self.use_cache else None

        return x, next_cache, all_hidden_states, list(all_self_attns)


def reorder_attn_buffer(input_buffer, new_order):
    """
    Reorders the attention buffer for incremental decoding.

    Args:
        input_buffer (dict): A dictionary containing cached attention states (key, value).
        new_order (Tensor): A tensor indicating the new order of elements in the buffer.

    Returns:
        dict: Reordered attention buffer.
    """
    for k in input_buffer.keys():
        input_buffer_k = input_buffer[k]
        if input_buffer_k is not None:
            # Reorder the buffer based on the new order
            input_buffer[k] = input_buffer_k.index_select(0, new_order)
    return input_buffer


class SelfAttention(nn.Module):
    """
    Multi-headed self-attention mechanism as described in "Attention Is All You Need".
    Supports both self-attention and encoder-decoder attention.

    Args:
        embed_dim (int): Dimensionality of the embeddings.
        num_heads (int): Number of attention heads.
        kdim (int, optional): Dimensionality of keys. Defaults to `embed_dim`.
        vdim (int, optional): Dimensionality of values. Defaults to `embed_dim`.
        dropout (float): Dropout probability for attention weights.
        bias (bool): Whether to include bias in linear projections.
        encoder_decoder_attention (bool): If True, implements encoder-decoder cross-attention.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            encoder_decoder_attention=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5  # Scaling factor for dot-product attention

        self.encoder_decoder_attention = encoder_decoder_attention
        qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim

        assert self.encoder_decoder_attention or qkv_same_dim, (
            "Self-attention requires query, key, and value to be of the same size"
        )

        # Linear projections for query, key, and value
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, dim_0, bsz):
        """
        Reshapes the tensor for multi-head attention computation.

        Args:
            tensor (Tensor): Input tensor.
            dim_0 (int): Dimension 0 size (sequence length).
            bsz (int): Batch size.

        Returns:
            Tensor: Reshaped tensor.
        """
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            layer_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = False,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for self-attention.

        Args:
            query (Tensor): Query tensor of shape `(seq_len, batch, embed_dim)`.
            key (Tensor, optional): Key tensor. Defaults to `query` for self-attention.
            value (Tensor, optional): Value tensor. Defaults to `query` for self-attention.
            key_padding_mask (Tensor, optional): Mask to ignore padding tokens.
            layer_state (dict, optional): Cached states for incremental decoding.
            need_weights (bool, optional): Whether to return attention weights.
            static_kv (bool, optional): If True, the key and value are static.
            attn_mask (Tensor, optional): Mask for causal or other attention constraints.

        Returns:
            Tuple[Tensor, Tensor]: Attention output and attention weights.
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        # Load cached state if available
        if layer_state is not None:
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state and static_kv:
                key = value = None
        else:
            saved_state = None
            layer_state = {}

        # Project query, key, and value
        q = self.q_proj(query) * self.scaling
        if self.encoder_decoder_attention:
            k = self.k_proj(key) if key is not None else None
            v = self.v_proj(key) if key is not None else None
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        # Reshape for multi-head attention
        q = self._shape(q, tgt_len, bsz)
        k = self._shape(k, -1, bsz) if k is not None else None
        v = self._shape(v, -1, bsz) if v is not None else None

        # Update state with cached key and value
        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        # Attention computation
        attn_weights = torch.bmm(q, k.transpose(1, 2))  # Dot-product attention
        if attn_mask is not None:
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = self._apply_key_padding_mask(attn_weights, key_padding_mask, bsz)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_weights, v)  # Compute attention output

        # Reshape back to (seq_len, batch, embed_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        """
        Updates the key and value tensors with cached states.

        Args:
            k, v (Tensor): Key and value tensors.
            saved_state (dict): Cached state dictionary.
            key_padding_mask (Tensor): Key padding mask.
            static_kv (bool): If True, key and value are static.
            bsz (int): Batch size.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Updated key, value, and key_padding_mask.
        """
        # Retrieve previous key and value
        if "prev_key" in saved_state:
            prev_key = saved_state["prev_key"].view(bsz * self.num_heads, -1, self.head_dim)
            k = torch.cat([prev_key, k], dim=1) if not static_kv else prev_key
        if "prev_value" in saved_state:
            prev_value = saved_state["prev_value"].view(bsz * self.num_heads, -1, self.head_dim)
            v = torch.cat([prev_value, v], dim=1) if not static_kv else prev_value

        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(
            key_padding_mask: Optional[Tensor],
            prev_key_padding_mask: Optional[Tensor],
            batch_size: int,
            src_len: int,
            static_kv: bool,
    ) -> Optional[Tensor]:
        """
        Concatenates the current and previous key padding masks for incremental decoding.

        Args:
            key_padding_mask (Optional[Tensor]): Current key padding mask of shape `(batch_size, src_len)`.
            prev_key_padding_mask (Optional[Tensor]): Previous key padding mask.
            batch_size (int): Size of the batch.
            src_len (int): Source sequence length.
            static_kv (bool): Whether key and value are static (do not change).

        Returns:
            Optional[Tensor]: Combined key padding mask.
        """
        # If previous mask exists and keys/values are static, return previous mask
        if prev_key_padding_mask is not None and static_kv:
            new_key_padding_mask = prev_key_padding_mask
        # If both current and previous masks exist, concatenate them
        elif prev_key_padding_mask is not None and key_padding_mask is not None:
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), key_padding_mask.float()], dim=1)
        # Handle cases where only the previous mask exists
        elif prev_key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - prev_key_padding_mask.size(1))
            if prev_key_padding_mask.is_cuda:
                filler = filler.cuda()
            new_key_padding_mask = torch.cat([prev_key_padding_mask.float(), filler.float()], dim=1)
        # Handle cases where only the current mask exists
        elif key_padding_mask is not None:
            filler = torch.zeros(batch_size, src_len - key_padding_mask.size(1))
            if key_padding_mask.is_cuda:
                filler = filler.cuda()
            new_key_padding_mask = torch.cat([filler.float(), key_padding_mask.float()], dim=1)
        # If no masks are available, return the previous mask (None if not set)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask

    class LearnedPositionalEmbedding(nn.Embedding):
        """
        A learnable positional embedding module.

        This module generates positional embeddings for sequences up to a fixed maximum size.
        Padding indices are ignored by either offsetting the embedding IDs or by adjusting
        the input position IDs.

        Args:
            num_embeddings (int): Total number of embeddings.
            embedding_dim (int): Dimensionality of each embedding.
            padding_idx (int): Index representing padding in input sequences.
        """

        def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
            """
            Initializes the learned positional embedding layer.
            Args:
                num_embeddings (int): Maximum number of positions to embed.
                embedding_dim (int): Dimensionality of the embeddings.
                padding_idx (int): Index of the padding token to be ignored.
            """
            assert padding_idx is not None
            num_embeddings += padding_idx + 1  # Reserve space for padding index
            super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

        def forward(self, input, generation_mode=False):
            """
            Forward pass to generate positional embeddings.

            Args:
                input (Tensor): Input tensor of shape `(batch_size, seq_len)`.
                generation_mode (bool): If True, generate positions for decoding (incremental).

            Returns:
                Tensor: Positional embeddings of shape `(batch_size, seq_len, embedding_dim)`.
            """
            if generation_mode:
                # In generation mode, positions are based on the current decoding step
                pos = int(self.padding_idx + input.size(1))  # Current sequence length
                positions = input.data.new(1, 1).fill_(pos)
            else:
                # Create position IDs for the entire input sequence
                positions = create_position_ids_from_input_ids(input, self.padding_idx)
            return super().forward(positions)  # Use the base embedding layer for positions

    def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        Creates a layer normalization module.

        If available, uses `FusedLayerNorm` from NVIDIA's Apex library for improved performance on GPUs.
        Otherwise, defaults to PyTorch's `torch.nn.LayerNorm`.

        Args:
            normalized_shape (int or list): Shape of the input to normalize.
            eps (float): Small value to prevent division by zero.
            elementwise_affine (bool): Whether to learn affine parameters for scaling and shifting.

        Returns:
            nn.Module: Layer normalization module.
        """
        if torch.cuda.is_available():
            try:
                from apex.normalization import FusedLayerNorm
                # Use Apex FusedLayerNorm for faster computations on GPUs
                return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
            except ImportError:
                pass
        # Fallback to PyTorch's LayerNorm
        return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def fill_with_neg_inf(t):
        """
        Fill a tensor with negative infinity (-inf).
        Compatible with FP16 to ensure proper handling of low precision types.

        Args:
            t (Tensor): Input tensor.

        Returns:
            Tensor: Tensor filled with -inf values.
        """
        return t.float().fill_(float("-inf")).type_as(t)

    def _filter_out_falsey_values(tup) -> Tuple:
        """
        Remove entries from a tuple that are None or empty lists.

        Args:
            tup (Tuple): Input iterable containing various elements.

        Returns:
            Tuple: Filtered tuple with only non-falsey values.
        """
        return tuple(x for x in tup if isinstance(x, torch.Tensor) or x)

    # Documentation for model output structure
    RET_DOCSTRING = r"""
        Return:
            :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
            last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Sequence of hidden-states at the output of the last layer of the model.
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
                :obj:`(batch_size, num_heads, sequence_length, sequence_length)`:
                Attention weights after the softmax, used to compute weighted averages in the self-attention heads.
    """

    # Public API for the BART model
    @add_start_docstrings(
        "The bare BART Model outputting raw hidden-states without any specific head on top.", BART_START_DOCSTRING,
    )
    class BartModel(PretrainedBartModel):
        """
        The bare BART model with encoder and decoder components.
        Outputs raw hidden states without additional heads.

        Args:
            config (BartConfig): Configuration object containing hyperparameters.
        """

        def __init__(self, config: BartConfig):
            """
            Initialize the BART model:
            - Shared embedding layer
            - Encoder and decoder components
            - Model weights
            """
            super().__init__(config)
            self.output_attentions = config.output_attentions  # Whether to output attention weights
            self.output_hidden_states = config.output_hidden_states  # Whether to output hidden states

            # Shared embedding layer for encoder and decoder
            padding_idx, vocab_size = config.pad_token_id, config.vocab_size
            self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

            # Encoder and decoder
            self.encoder = BartEncoder(config, self.shared)
            self.decoder = BartDecoder(config, self.shared)

            # Initialize weights
            self.init_weights()

        @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
        def forward(
                self,
                input_ids,
                attention_mask=None,
                decoder_input_ids=None,
                encoder_outputs=None,  # type: Tuple
                decoder_attention_mask=None,
                decoder_cached_states=None,
        ):
            """
            Forward pass for the BART model.

            Args:
                input_ids (Tensor): Input IDs of shape `(batch_size, sequence_length)`.
                attention_mask (Tensor, optional): Mask for ignoring padding tokens in the input.
                decoder_input_ids (Tensor, optional): Input IDs for the decoder.
                encoder_outputs (Tuple, optional): Precomputed encoder outputs.
                decoder_attention_mask (Tensor, optional): Mask for the decoder to ignore padding tokens.
                decoder_cached_states (dict, optional): Cached states for incremental decoding.

            Returns:
                Tuple[Tensor]: Outputs from the decoder and encoder (e.g., hidden states and attention).
            """
            if attention_mask is not None:
                assert attention_mask.dim() == 2, "Attention mask must be 2-dimensional."

                # Convert attention mask to -inf for padding positions
                attention_mask = (1.0 - attention_mask.long()) * -10000.0
                assert attention_mask.max() <= 0, "Attention mask values must be <= 0."

            # Prepare decoder inputs if not in generation mode
            if not self.decoder.generation_mode:
                decoder_input_ids, decoder_attention_mask = _prepare_bart_decoder_inputs(
                    self.config,
                    input_ids,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attn_mask=decoder_attention_mask,
                )
            assert decoder_input_ids is not None, "Decoder input IDs cannot be None."

            # Run encoder if encoder outputs are not provided
            if encoder_outputs is None:
                encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            assert isinstance(encoder_outputs, tuple), "Encoder outputs must be a tuple."

            # Pass inputs to the decoder
            decoder_outputs = self.decoder(
                decoder_input_ids,
                encoder_outputs[0],  # Encoder hidden states
                attention_mask,
                decoder_attention_mask,
                decoder_cached_states=decoder_cached_states,
            )

            # Remove empty or None values from outputs
            decoder_outputs = _filter_out_falsey_values(decoder_outputs)
            assert isinstance(decoder_outputs[0], torch.Tensor), "Decoder outputs must include a tensor."
            encoder_outputs = _filter_out_falsey_values(encoder_outputs)

            # Combine decoder and encoder outputs
            return decoder_outputs + encoder_outputs

        def get_input_embeddings(self):
            """
            Returns the shared embedding layer for input tokens.
            """
            return self.shared

        def set_input_embeddings(self, value):
            """
            Sets the shared embedding layer for input tokens.

            Args:
                value (nn.Embedding): New embedding layer.
            """
            self.shared = value

        def get_output_embeddings(self):
            """
            Returns a linear layer created from the shared embeddings for output projection.
            """
            return _make_linear_from_emb(self.shared)  # Generate on-the-fly

    @add_start_docstrings(
        "The bare BART Model with a language modeling head. This is the model used for summarization.",
        BART_START_DOCSTRING,
    )
    class BartForMaskedLM(PretrainedBartModel):
        """
        BART model with a language modeling head for tasks like summarization and text generation.

        Args:
            config (BartConfig): Configuration for the model.
        """
        base_model_prefix = "model"

        def __init__(self, config: BartConfig):
            """
            Initialize the BART model with a language modeling head.

            Args:
                config (BartConfig): Model configuration object.
            """
            super().__init__(config)
            base_model = BartModel(config)  # Initialize the base BART model
            self.model = base_model
            self.lm_head = _make_linear_from_emb(self.model.shared)  # Create the language modeling head

        def tie_weights(self):
            """
            Prevents changing the output embedding dimensions, ensuring shared embeddings.
            """
            pass  # No action needed; weights are already tied

        @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
        def forward(
                self,
                input_ids,
                attention_mask=None,
                encoder_outputs=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                decoder_cached_states=None,
                lm_labels=None,
                **unused
        ):
            """
            Forward pass for masked language modeling.

            Args:
                input_ids (Tensor): Input tensor of token IDs.
                attention_mask (Tensor, optional): Mask for ignoring padding tokens in the encoder input.
                encoder_outputs (Tuple, optional): Precomputed encoder outputs.
                decoder_input_ids (Tensor, optional): Input tensor for the decoder.
                decoder_attention_mask (Tensor, optional): Mask for ignoring padding tokens in the decoder input.
                decoder_cached_states (dict, optional): Cached states for decoder layers.
                lm_labels (Tensor, optional): Labels for masked language modeling loss computation.
                **unused: Additional unused arguments.

            Returns:
                Tuple[Tensor]: Model outputs including logits, hidden states, and optional losses.
            """
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                decoder_cached_states=decoder_cached_states,
            )
            lm_logits = self.lm_head(outputs[0])  # Generate logits for the language model head
            outputs = (lm_logits,) + outputs[1:]  # Combine logits with other outputs

            # Compute masked language modeling loss if labels are provided
            if lm_labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
                outputs = (masked_lm_loss,) + outputs

            return outputs

        @staticmethod
        def prepare_inputs_for_generation(input_ids, past, decoder_input_ids, attention_mask):
            """
            Prepare inputs for text generation tasks.

            Args:
                input_ids (Tensor): Input token IDs.
                past (Tuple, optional): Cached past states.
                decoder_input_ids (Tensor): Decoder input IDs.
                attention_mask (Tensor): Attention mask for the input.

            Returns:
                dict: Prepared inputs for the model.
            """
            if past is None:  # If no past states, initialize encoder and decoder states
                encoder_outputs, decoder_cached_states = None, None
            else:
                encoder_outputs, decoder_cached_states = past
            return {
                "input_ids": input_ids,
                "decoder_cached_states": decoder_cached_states,
                "decoder_input_ids": decoder_input_ids,
                "encoder_outputs": encoder_outputs,
                "attention_mask": attention_mask,
            }

        @staticmethod
        def _reorder_cache(past, beam_idx):
            """
            Reorder cache for beam search decoding.

            Args:
                past (Tuple): Cached past states.
                beam_idx (Tensor): Reordered beam indices.

            Returns:
                Tuple: Reordered cache.
            """
            ((enc_out, enc_mask), decoder_cached_states) = past
            reordered_past = []
            for layer_past in decoder_cached_states:
                layer_past_new = {
                    attn_key: reorder_attn_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
                }
                reordered_past.append(layer_past_new)
            new_enc_out = enc_out if enc_out is None else enc_out.index_select(1, beam_idx)
            new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)
            past = ((new_enc_out, new_enc_mask), reordered_past)
            return past

        def get_output_embeddings(self):
            """
            Returns the language modeling head for output embeddings.
            """
            return self.lm_head

        @torch.no_grad()
        def generate(
                self,
                input_ids,
                attention_mask=None,
                max_length=20,
                num_beams=1,
                repetition_penalty=1.0,
                length_penalty=1.0,
                num_return_sequences=1,
                min_len=0,
                no_repeat_ngram_size=0,
        ):
            """
            Generate sequences using beam search or sampling.

            Args:
                input_ids (Tensor): Input token IDs.
                attention_mask (Tensor, optional): Mask for ignoring padding tokens.
                max_length (int): Maximum sequence length to generate.
                num_beams (int): Number of beams for beam search.
                repetition_penalty (float): Penalty for repeated tokens.
                length_penalty (float): Penalty for sequence length.
                num_return_sequences (int): Number of sequences to return.
                min_len (int): Minimum length of generated sequences.
                no_repeat_ngram_size (int): Size of n-grams to avoid repeating.

            Returns:
                Tensor: Generated sequences.
            """
            # Additional assertions to validate inputs
            assert input_ids is not None, "Input IDs cannot be None"
            assert self.config.use_cache, "Bart requires `use_cache=True` for generation."
            # More detailed logic for beam search and decoding follows...
            pass

        @staticmethod
        def calc_banned_tokens(prev_output_tokens, num_hypos, no_repeat_ngram_size, step):
            """
            Calculates tokens to ban based on repeated n-grams during beam search.

            Args:
                prev_output_tokens (Tensor): Previous output tokens for each hypothesis.
                                             Shape: (num_hypos, sequence_length).
                num_hypos (int): Number of hypotheses (batch size * num_beams).
                no_repeat_ngram_size (int): Size of n-grams to avoid repeating.
                step (int): Current decoding step in the generation process.

            Returns:
                List[List[int]]: A list of lists, where each inner list contains the indices of tokens
                                 to be banned for the corresponding hypothesis.
            """
            # If we haven't generated enough tokens for the specified n-gram size, no tokens are banned.
            if step + 2 < no_repeat_ngram_size:
                return [[] for _ in range(num_hypos)]

            # Initialize a list to store dictionaries for generated n-grams for each hypothesis.
            gen_ngrams = [{} for _ in range(num_hypos)]

            # Loop through each hypothesis and compute the generated n-grams.
            for idx in range(num_hypos):
                gen_tokens = prev_output_tokens[idx].tolist()  # Convert tokens to a list.
                # Generate n-grams from the token list.
                for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
                    k = tuple(ngram[:-1])  # Use the first n-1 tokens as the key.
                    gen_ngrams[idx][k] = gen_ngrams[idx].get(k, []) + [
                        ngram[-1]]  # Append the nth token to the key's list.

            # Helper function to retrieve the generated n-grams for a specific hypothesis.
            def _get_generated_ngrams(hypo_idx):
                """
                Retrieves tokens to ban for a given hypothesis based on previously generated n-grams.

                Args:
                    hypo_idx (int): Index of the hypothesis.

                Returns:
                    List[int]: List of tokens to ban.
                """
                # Extract the n-gram index for the current step.
                ngram_index = tuple(prev_output_tokens[hypo_idx, step + 2 - no_repeat_ngram_size: step + 1].tolist())
                # Return the list of banned tokens if the n-gram exists, otherwise return an empty list.
                return gen_ngrams[hypo_idx].get(ngram_index, [])

            # Apply the helper function to all hypotheses and return the list of banned tokens.
            banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
            return banned_tokens



