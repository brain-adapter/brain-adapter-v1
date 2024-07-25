from typing import Optional, Tuple, Type
import torch
from torch import nn
from omegaconf import DictConfig
import torch.nn.functional as F

from .activations import get_activation


def get_class(name: str) -> Type:
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [
                torch.Generator(device).manual_seed(seed_item) for seed_item in seed
            ]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def get_device(device_name: Optional[str]) -> torch.device:
    if device_name is not None:
        return torch.device(device_name)
    return (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )


class EEGEmbeddings(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_samples: int,
        patch_size: int,
        embed_dim: int,
    ):
        super().__init__()
        assert (
            num_samples % patch_size == 0
        ), "[num_samples] should be divisible by [patch_size]"
        self.patch_embedding = nn.Conv1d(
            in_channels=num_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        self.class_embedding = nn.Parameter(torch.randn(embed_dim))

        self.num_patches = num_samples // patch_size
        self.num_positions = self.num_patches + 1
        self.embed_dim = embed_dim

        self.position_embedding = nn.Embedding(self.num_positions, embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, values: torch.Tensor):
        batch_size = values.shape[0]
        patch_embeds = (
            self.patch_embedding(values).transpose(1, 2).contiguous()
        )  # shape = [batch_size, num_sequence, embed_dim]

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float,
        act_fn: str,
        dropout: float,
    ):
        super().__init__()
        self.activation_fn = get_activation(act_fn)
        self.intermediate_size = int(hidden_size * mlp_ratio)
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(hidden_size, self.intermediate_size)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(self.intermediate_size, hidden_size)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.drop1(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = self.drop2(hidden_states)
        return hidden_states


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float,
        mlp_ratio: float,
        act_fn: str,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = MLP(
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            act_fn=act_fn,
            dropout=dropout,
        )
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
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_dim: int = None,
        dropout: float = 0.0,
        mlp_ratio: float = 1.0,
        act_fn: str = "gelu",
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.kv_dim = kv_dim if kv_dim is not None else hidden_size

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            kdim=self.kv_dim,
            vdim=self.kv_dim,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(self.embed_dim)
        self.kv_norm = nn.LayerNorm(self.kv_dim)

        self.mlp = MLP(
            hidden_size=hidden_size,
            mlp_ratio=mlp_ratio,
            act_fn=act_fn,
            dropout=dropout,
        )
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
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        dropout: Optional[float] = None,
        mlp_ratio: Optional[float] = None,
        act_fn: Optional[str] = None,
    ):
        super().__init__()

        dropout = dropout if dropout is not None else 0.0
        mlp_ratio = mlp_ratio if mlp_ratio is not None else 1.0
        act_fn = act_fn if act_fn is not None else "gelu"

        self.layers = nn.ModuleList(
            [
                SelfAttentionLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    act_fn=act_fn,
                )
                for _ in range(num_layers)
            ]
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
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        kv_dim: int = None,
        dropout: Optional[float] = None,
        mlp_ratio: Optional[float] = None,
        act_fn: Optional[str] = None,
    ):
        super().__init__()

        dropout = dropout if dropout is not None else 0.0
        mlp_ratio = mlp_ratio if mlp_ratio is not None else 4.0
        act_fn = act_fn if act_fn is not None else "gelu"

        self.layers = nn.ModuleList(
            [
                CrossAttentionLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    kv_dim=kv_dim,
                    dropout=dropout,
                    mlp_ratio=mlp_ratio,
                    act_fn=act_fn,
                )
                for _ in range(num_layers)
            ]
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
        # learnable pos embedding is more compatible for EEG features
        self.embeddings = EEGEmbeddings(
            num_channels=config.num_channels,
            num_samples=config.num_samples,
            patch_size=config.patch_size,
            embed_dim=config.hidden_size,
        )
        self.pre_layernorm = nn.LayerNorm(config.hidden_size)

        self.block = SelfAttentionBlock(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout=config.get("dropout", None),
            mlp_ratio=config.get("mlp_ratio", None),
            act_fn=config.get("act_fn", None),
        )

        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        eeg_values: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        hidden_states = self.embeddings(eeg_values)

        encoder_outputs: Tuple = self.block(
            hidden_states, output_attentions=output_attentions
        )

        last_hidden_state: torch.Tensor = encoder_outputs[0]

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return (pooled_output,) + encoder_outputs[1:]


class VisionPerceiver(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.input_proj = nn.Linear(config.input_dim, config.hidden_size)

        self.register_buffer(
            "position_ids",
            torch.arange(config.num_tokens).expand((1, -1)),
            persistent=False,
        )
        self.position_embedding = nn.Embedding(config.num_tokens, config.input_dim)

        self.latents = nn.Parameter(torch.randn(config.hidden_size))

        self.block = CrossAttentionBlock(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout=config.get("dropout", 0.0),
            mlp_ratio=config.get("mlp_ratio", 4.0),
            act_fn=config.get("act_fn", "gelu"),
        )

        self.post_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        tokens: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        batch_size = tokens.shape[0]

        tokens = tokens + self.position_embedding(self.position_ids)

        latents = self.latents.expand(batch_size, 1, -1)
        tokens = self.input_proj(tokens)

        encoder_outputs: Tuple = self.block(
            latents, tokens, output_attentions=output_attentions
        )

        last_hidden_state: torch.Tensor = encoder_outputs[0]

        pooled_output = last_hidden_state.squeeze(dim=1)
        pooled_output = self.post_layernorm(pooled_output)

        return (pooled_output,) + encoder_outputs[1:]


class AdapterProjection(nn.Module):
    def __init__(
        self,
        input_size: int,
        cross_attention_dim: int = 768,
        num_tokens: int = 4,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim

        self.proj = nn.Linear(input_size, num_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

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

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
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
        hidden_states = F.scaled_dot_product_attention(
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


class VisionAttnProcessor(nn.Module):
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
    ):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens if num_tokens is not None else 4
        self.scale = nn.Parameter(torch.tensor(0.0))

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
        hidden_states = F.scaled_dot_product_attention(
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
        ip_hidden_states = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        with torch.no_grad():
            self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
            # print(self.attn_map.shape)

        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        ip_hidden_states = ip_hidden_states.to(query.dtype)

        hidden_states = hidden_states + self.scale.tanh() * ip_hidden_states

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
