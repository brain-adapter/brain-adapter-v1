import torch
from torch import nn
from omegaconf import DictConfig
from typing import Optional, Tuple, Union
from diffusers.models.attention_processor import Attention

from .activations import get_activation


class EEGEmbeddings(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        assert (
            config.num_samples % config.patch_size == 0
        ), "[num_samples] should be divisible by [patch_size]"

        self.embed_dim: int = config.hidden_size

        self.patch_embedding = nn.Conv1d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        self.num_patches = config.num_samples // config.patch_size
        self.num_positions = self.num_patches

        if config.pool_type == "cls":
            self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))
            self.num_positions = self.num_positions + 1
        else:
            self.register_parameter("class_embedding", None)

        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, values: torch.Tensor):
        batch_size = values.shape[0]

        embeds = []

        if self.class_embedding is not None:
            class_embeds = self.class_embedding.expand(batch_size, 1, -1)
            embeds.append(class_embeds)

        patch_embeds = (
            self.patch_embedding(values).transpose(1, 2).contiguous()
        )  # shape = [batch_size, num_sequence, embed_dim]

        embeds.append(patch_embeds)

        embeds = torch.cat(embeds, dim=1)

        embeds = embeds + self.position_embedding(self.position_ids)
        return embeds


class EEGEmebeddingsWithMOE(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        assert (
            config.num_samples % config.patch_size == 0
        ), "[num_samples] should be divisible by [patch_size]"

        self.embed_dim: int = config.hidden_size

        self.patch_embedding = nn.Conv1d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        self.num_patches = config.num_samples // config.patch_size
        self.num_positions = self.num_patches + 1

        # For each subject, there will be a class embedding, but problem also occurs when coming across with
        #  unseen subjects.
        self.class_embedding = nn.ParameterList(
            torch.randn(self.embed_dim) for _ in range(config.num_subjects)
        )

        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, values: torch.Tensor, subjects: torch.Tensor):
        class_embeds = torch.stack(
            [self.class_embedding[s] for s in subjects]
        ).unsqueeze(dim=1)

        patch_embeds = (
            self.patch_embedding(values).transpose(1, 2).contiguous()
        )  # shape = [batch_size, num_sequence, embed_dim]

        embeds = torch.cat([class_embeds, patch_embeds], dim=1)

        embeds = embeds + self.position_embedding(self.position_ids)
        return embeds


class MLP(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.intermediate_size = int(config.hidden_size * config.mlp_ratio)
        self.hidden_size = config.hidden_size

        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.intermediate_size),
            get_activation(config.act_fn),
            nn.Dropout(config.dropout),
            nn.Linear(self.intermediate_size, self.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states)


class SelfAttentionLayer(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self, hidden_states: torch.Tensor, output_attentions: Optional[bool] = False
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            need_weights=output_attentions,
            average_attn_weights=False,
        )
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class CrossAttentionLayer(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.embed_dim: int = config.hidden_size
        self.kv_dim: int = config.kv_dim

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            kdim=self.kv_dim,
            vdim=self.kv_dim,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(self.embed_dim)
        self.kv_norm = nn.LayerNorm(self.kv_dim)

        self.mlp = MLP(config)
        self.mlp_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = query

        query = self.query_norm(query)
        key_value = self.kv_norm(key_value)
        hidden_states, attn_weights = self.cross_attn(
            query=query, key=key_value, value=key_value
        )
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class SelfAttentionBlock(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.layers = nn.ModuleList(
            [SelfAttentionLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
        self, inputs_embeds: torch.Tensor, output_attentions: Optional[bool] = None
    ) -> Tuple[torch.FloatTensor]:
        output_attentions = (
            output_attentions if output_attentions is not None else False
        )
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        return tuple(v for v in [hidden_states, all_attentions] if v is not None)


class CrossAttentionBlock(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.layers = nn.ModuleList(
            [CrossAttentionLayer(config) for _ in range(config.num_layers)]
        )

    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        all_attentions = () if output_attentions else None

        hidden_states = latents
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                tokens,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        return tuple(v for v in [hidden_states, all_attentions] if v is not None)


class EEGTransformer(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        self.pool_type = config.get("pool_type", "cls")

        # learnable pos embedding is more compatible for EEG features
        self.embeddings = EEGEmebeddingsWithMOE(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size)

        self.block = SelfAttentionBlock(config)

        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        eeg_values: torch.Tensor,
        subjects: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.FloatTensor, Tuple[torch.FloatTensor]]]:
        hidden_states = self.embeddings(eeg_values, subjects - 1)

        encoder_outputs: Tuple = self.block(
            hidden_states, output_attentions=output_attentions
        )

        last_hidden_state: torch.Tensor = encoder_outputs[0]

        if self.pool_type == "cls":
            pooled_output = last_hidden_state[:, 0, :]
        elif self.pool_type == "avg":
            pooled_output = torch.mean(last_hidden_state, dim=1)
        else:
            pooled_output = last_hidden_state

        pooled_output = self.post_layernorm(pooled_output)

        return (
            pooled_output,
            last_hidden_state,
        ) + encoder_outputs[1:]


class PerceiverResampler(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.input_dim, config.cross_attention_dim)

        self.latents = nn.Parameter(
            torch.randn(1, config.query_tokens, config.hidden_size)
            / config.hidden_size**0.5
        )

        self.block = CrossAttentionBlock(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            kv_dim=config.cross_attention_dim,
            num_attention_heads=config.num_attention_heads,
            dropout=config.get("dropout", 0.0),
            mlp_ratio=config.get("mlp_ratio", 1.0),
            act_fn=config.get("act_fn", "gelu"),
        )

        self.output_proj = nn.Linear(config.hidden_size, config.output_dim)

        self.post_layernorm = nn.LayerNorm(config.output_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        batch_size = tokens.shape[0]

        latents = self.latents.repeat(batch_size, 1, 1)
        tokens = self.input_proj(tokens)

        encoder_outputs: Tuple = self.block(
            latents, tokens, output_attentions=output_attentions
        )

        last_hidden_state = encoder_outputs[0]

        pooled_output: torch.Tensor = self.output_proj(last_hidden_state)
        pooled_output = self.post_layernorm(pooled_output)

        return (pooled_output,) + encoder_outputs[1:]


class AdapterProjection(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.num_tokens: int = config.num_tokens
        self.cross_attention_dim = config.cross_attention_dim
        self.input_dim = config.input_dim

        self.proj = nn.Linear(
            self.input_dim, self.num_tokens * self.cross_attention_dim
        )
        self.norm = nn.LayerNorm(self.cross_attention_dim)

    def forward(self, condition_embeds: torch.Tensor):
        context_tokens = self.proj(condition_embeds).reshape(
            -1, self.num_tokens, self.cross_attention_dim
        )

        norm_tokens = self.norm(context_tokens)

        return norm_tokens


class AttnProcessor(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        super().__init__()
        if not hasattr(nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for IP-Adapater for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, defaults to 4 when do ip_adapter_plus it should be 16):
            The context length of the image features.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        num_tokens: Optional[int] = None,
        scale: Optional[float] = None,
    ):
        super().__init__()

        if not hasattr(nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens if num_tokens is not None else 4
        self.scale = scale if scale is not None else 1.0

        self.to_k_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )
        self.to_v_ip = nn.Linear(
            cross_attention_dim or hidden_size, hidden_size, bias=False
        )

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        else:
            # get encoder_hidden_states, ip_hidden_states
            end_pos = encoder_hidden_states.shape[1] - self.num_tokens
            encoder_hidden_states, ip_hidden_states = (
                encoder_hidden_states[:, :end_pos, :],
                encoder_hidden_states[:, end_pos:, :],
            )
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # for ip-adapter
        ip_key = self.to_k_ip(ip_hidden_states)
        ip_value = self.to_v_ip(ip_hidden_states)

        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        ip_hidden_states = nn.functional.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        with torch.no_grad():
            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
            # print(self.attn_map.shape)

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + self.scale * ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class VisionAttnProcessor(nn.Module):
    r"""
    Attention processor for vision-only condition for PyTorch 2.0.
    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
    """

    def __init__(self, hidden_size: int, cross_attention_dim: int):
        super().__init__()

        if not hasattr(nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        self.to_key = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_value = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = encoder_hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        key = self.to_key(encoder_hidden_states)
        value = self.to_value(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class MixedAttnProcessor(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_vision_tokens: int = 4,
    ):
        super().__init__()

        if not hasattr(nn.functional, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_vision_tokens = num_vision_tokens

        self.to_key_eeg = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_value_eeg = nn.Linear(cross_attention_dim, hidden_size, bias=False)

        self.to_key_vision = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_value_vision = nn.Linear(cross_attention_dim, hidden_size, bias=False)

        self.eeg_scale = nn.Parameter(torch.tensor(0.0))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query: torch.Tensor = attn.to_q(hidden_states)

        vision_tokens: torch.Tensor = encoder_hidden_states[:, : self.num_vision_tokens]
        vision_key: torch.Tensor = self.to_key_vision(vision_tokens)
        vision_value: torch.Tensor = self.to_value_vision(vision_tokens)

        head_dim = self.hidden_size // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        vision_key = vision_key.view(batch_size, -1, attn.heads, head_dim).transpose(
            1, 2
        )
        vision_value = vision_value.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = nn.functional.scaled_dot_product_attention(
            query,
            vision_key,
            vision_value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        eeg_tokens: torch.Tensor = encoder_hidden_states[:, self.num_vision_tokens :]
        eeg_key: torch.Tensor = self.to_key_eeg(eeg_tokens)
        eeg_value: torch.Tensor = self.to_value_eeg(eeg_tokens)

        eeg_key = eeg_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        eeg_value = eeg_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        eeg_hidden_states: torch.Tensor = nn.functional.scaled_dot_product_attention(
            query, eeg_key, eeg_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )

        eeg_hidden_states = eeg_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        eeg_hidden_states = eeg_hidden_states.to(query.dtype)

        hidden_states = hidden_states + eeg_hidden_states * self.eeg_scale.tanh()

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
