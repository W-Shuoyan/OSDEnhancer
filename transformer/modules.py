from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import deprecate
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.utils.import_utils import is_torch_version
from diffusers.models.attention import Attention
from diffusers.models.normalization import CogVideoXLayerNormZero

class CogVideoXPatchResEmbed(nn.Module):
    """
    3D Residual patch embedding for CogVideoX.

    Input:  (B, F, C, H, W)
    Output: (B, num_patches_3d, inner_dim)
    """

    def __init__(
        self,
        patch_size: int = 2,
        patch_size_t: int = 2,
        in_channels: int = 16,
        dim_head: int = None,
        heads: int = None,
    ) -> None:
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        if patch_size_t is None:
            self.patch_size_t = 1

        super().__init__()

        self.res_proj = nn.Linear(
            in_channels * patch_size * patch_size * self.patch_size_t,
            dim_head * heads
        )

        self.res_ff0_proj = nn.Linear(
            in_channels * patch_size * patch_size * self.patch_size_t,
            dim_head * heads * 4
        )

    def forward(self, res: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = res.shape

        p = self.patch_size
        p_t = self.patch_size_t

        x = torch.abs(res.permute(0, 1, 3, 4, 2).clamp(-1, 1))

        x = x.reshape(
            batch_size,
            num_frames // p_t, p_t,
            height // p, p,
            width // p, p,
            channels,
        )

        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)

        x = x.flatten(4, 7)

        x = x.flatten(1, 3)

        res_hidden_states = self.res_proj(x)
        res_ff0_hidden_states = self.res_ff0_proj(x)       

        return res_hidden_states.contiguous(), res_ff0_hidden_states.contiguous()
    

class CogVideoXAttnProcessor2_0_TCLoRA:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,                # image tokens (before cat)
        encoder_hidden_states: torch.Tensor,        # text tokens
        res_hidden_states: torch.Tensor,     # (B, image_len, 1 or C), aligned with image part
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # text length is determined by encoder_hidden_states
        text_seq_length = encoder_hidden_states.size(1)

        # concat text + image
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # ----------------- Q / K / V with image-only LoRA -----------------
        # base projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # only image part participates in LoRA; text part remains unchanged
        # image token slice
        image_slice = slice(text_seq_length, sequence_length)

        # LoRA on Q
        if hasattr(attn, "to_q_TC_lora") and attn.to_q_TC_lora is not None and res_hidden_states is not None:
            image_hidden = hidden_states[:, image_slice, :]  # (B, image_len, C_in)
            delta_q_img = attn.to_q_TC_lora(image_hidden)       # (B, image_len, C_out)

            # scale by res_hidden_states; assume broadcastable
            delta_q_img = delta_q_img * res_hidden_states

            # build full delta with zeros on text part
            zero_text = torch.zeros(
                batch_size,
                text_seq_length,
                delta_q_img.size(-1),
                device=delta_q_img.device,
                dtype=delta_q_img.dtype,
            )
            delta_q = torch.cat([zero_text, delta_q_img], dim=1)
            query = query + delta_q

        # LoRA on K
        if hasattr(attn, "to_k_TC_lora") and attn.to_k_TC_lora is not None and res_hidden_states is not None:
            image_hidden = hidden_states[:, image_slice, :]
            delta_k_img = attn.to_k_TC_lora(image_hidden)
            delta_k_img = delta_k_img * res_hidden_states

            zero_text = torch.zeros(
                batch_size,
                text_seq_length,
                delta_k_img.size(-1),
                device=delta_k_img.device,
                dtype=delta_k_img.dtype,
            )
            delta_k = torch.cat([zero_text, delta_k_img], dim=1)
            key = key + delta_k

        # LoRA on V
        if hasattr(attn, "to_v_TC_lora") and attn.to_v_TC_lora is not None and res_hidden_states is not None:
            image_hidden = hidden_states[:, image_slice, :]
            delta_v_img = attn.to_v_TC_lora(image_hidden)
            delta_v_img = delta_v_img * res_hidden_states

            zero_text = torch.zeros(
                batch_size,
                text_seq_length,
                delta_v_img.size(-1),
                device=delta_v_img.device,
                dtype=delta_v_img.dtype,
            )
            delta_v = torch.cat([zero_text, delta_v_img], dim=1)
            value = value + delta_v

        # ----------------- reshape to heads -----------------
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ----------------- RoPE on image tokens -----------------
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            # only apply to image tokens (after text_seq_length)
            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        # ----------------- attention -----------------
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        # ----------------- to_out[0] + LoRA (image only) -----------------
        base_out = attn.to_out[0](hidden_states)

        if hasattr(attn, "to_out_TC_lora") and attn.to_out_TC_lora is not None and res_hidden_states is not None:
            image_hidden = hidden_states[:, image_slice, :]          # (B, image_len, C_out_in)
            delta_out_img = attn.to_out_TC_lora(image_hidden)           # (B, image_len, C_out)

            # scale by res_hidden_states; assume broadcastable
            delta_out_img = delta_out_img * res_hidden_states

            zero_text = torch.zeros(
                batch_size,
                text_seq_length,
                delta_out_img.size(-1),
                device=delta_out_img.device,
                dtype=delta_out_img.dtype,
            )
            delta_out = torch.cat([zero_text, delta_out_img], dim=1)
            hidden_states = base_out + delta_out
        else:
            hidden_states = base_out

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # ----------------- split back to text / image -----------------
        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states
    
class Linear_TCLoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 4.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0

        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)

            # common LoRA init: A random, B zeros -> initial delta = 0
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
            nn.init.zeros_(self.lora_B.weight)
        else:
            # degenerate case: no LoRA
            self.lora_A = None
            self.lora_B = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., in_features)
        return: (..., out_features)
        """
        if self.r == 0 or self.lora_A is None or self.lora_B is None:
            # no-op if rank = 0
            return torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)

        orig_shape = x.shape  # (B, L, C) or any shape with last dim = in_features
        x_2d = x.reshape(-1, self.in_features)
        delta = self.lora_B(self.lora_A(x_2d)) * self.scaling  # (N, out_features)
        delta = delta.view(*orig_shape[:-1], self.out_features)
        return delta

class GELU_TCLoRA(nn.Module):
    r"""
    GELU activation with optional LoRA on the input projection.

    Parameters:
        dim_in (`int`): Input channels.
        dim_out (`int`): Output channels.
        approximate (`str`): GELU approximation.
        bias (`bool`): Use bias in the base projection.
        text_seq_length (`int`): Number of text tokens at the head of the sequence.
        r (`int`): LoRA rank.
        alpha (`float`): LoRA alpha scaling.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        bias: bool = True,
        r: Optional[int] = None,
        alpha: Optional[float] = None,
    ):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

        # LoRA delta module for proj
        self.proj_TC_lora = Linear_TCLoRA(
            in_features=dim_in,
            out_features=dim_out,
            r=int(r) if r is not None else 0,
            alpha=float(alpha) if alpha is not None else 1.0,
        )

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type == "mps" and is_torch_version("<", "2.0.0"):
            # fp16 gelu not supported on mps before torch 2.0
            return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)
        return F.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states: torch.Tensor, res_hidden_states: torch.Tensor, text_seq_length: int):
        """
        hidden_states: (B, L, C_in)
        res_hidden_states: (B, L_img, C_out or 1),
        """
        base = self.proj(hidden_states)

        if self.proj_TC_lora.r == 0 or res_hidden_states is None:
            return self.gelu(base)

        B, L, _ = hidden_states.shape
        img_len = L - text_seq_length
        assert img_len > 0, "No image tokens found for LoRA in GELU."

        assert res_hidden_states.shape[0] == B, "Batch size mismatch for res_hidden_states."
        assert res_hidden_states.shape[1] == img_len, "Sequence length of res_hidden_states must equal image tokens."

        image_slice = slice(text_seq_length, L)

        image_hidden = hidden_states[:, image_slice, :]       # (B, img_len, C_in)
        delta_img = self.proj_TC_lora(image_hidden)              # (B, img_len, C_out)
        delta_img = delta_img * res_hidden_states

        zero_text = torch.zeros(
            B, text_seq_length, delta_img.size(-1),
            device=delta_img.device,
            dtype=delta_img.dtype,
        )
        delta = torch.cat([zero_text, delta_img], dim=1)      # (B, L, C_out)

        hidden_states = base + delta
        hidden_states = self.gelu(hidden_states)
        return hidden_states
    
class FeedForward_TCLoRA(nn.Module):
    r"""
    A feed-forward layer with LoRA on net[0].proj and net[2],
    applied only on image tokens and scaled by res_ff_hidden_sattes.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
        r: Optional[int] = None,
        alpha: Optional[float] = None,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        self.lora_rank = int(r) if r is not None else 0
        self.lora_alpha = float(alpha) if alpha is not None else 1.0

        self.net = nn.ModuleList([])

        # 0. project in + GELU
        self.net.append(
            GELU_TCLoRA(
                dim_in=dim,
                dim_out=inner_dim,
                approximate="tanh",
                bias=bias,
                r=self.lora_rank,
                alpha=self.lora_alpha,
            )
        )
        # 1. dropout
        self.net.append(nn.Dropout(dropout))
        # 2. project out (base Linear)
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))

        self.proj_out_TC_lora = Linear_TCLoRA(
            in_features=inner_dim,
            out_features=dim_out,
            r=self.lora_rank,
            alpha=self.lora_alpha,
        )

        self.final_dropout = final_dropout
        if self.final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, res_ff0_hidden_sattes: torch.Tensor, res_hidden_states: torch.Tensor, text_seq_length: int, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = (
                "The `scale` argument is deprecated and will be ignored. Please remove it, "
                "as passing it will raise an error in the future. `scale` should directly be "
                "passed while calling the underlying pipeline component i.e., via "
                "`cross_attention_kwargs`."
            )
            deprecate("scale", "1.0.0", deprecation_message)

        hidden_states = self.net[0](hidden_states, res_ff0_hidden_sattes, text_seq_length)

        hidden_states = self.net[1](hidden_states)

        base_out = self.net[2](hidden_states)

        if self.proj_out_TC_lora.r > 0 and res_hidden_states is not None:
            B, L, _ = hidden_states.shape
            img_len = L - text_seq_length
            assert img_len > 0, "No image tokens found for LoRA in FeedForward net[2]."

            assert res_hidden_states.shape[0] == B, "Batch size mismatch for res_ff_hidden_sattes."
            assert res_hidden_states.shape[1] == img_len, "Sequence length mismatch for image tokens."

            image_slice = slice(text_seq_length, L)

            image_hidden = hidden_states[:, image_slice, :]          # (B, img_len, inner_dim)
            delta_img = self.proj_out_TC_lora(image_hidden)             # (B, img_len, dim_out)

            delta_img = delta_img * res_hidden_states          # broadcast or exact match

            zero_text = torch.zeros(
                B, text_seq_length, delta_img.size(-1),
                device=delta_img.device,
                dtype=delta_img.dtype,
            )
            delta = torch.cat([zero_text, delta_img], dim=1)         # (B, L, dim_out)

            hidden_states = base_out + delta
        else:
            hidden_states = base_out

        if self.final_dropout:
            hidden_states = self.net[-1](hidden_states)

        return hidden_states
    
    
@maybe_allow_in_graph
class CogVideoX_TCLoRA_Block(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        # TC-LoRA hyper-parameters
        TC_lora_rank: Optional[int] = None,
        TC_lora_alpha: Optional[float] = None,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0_TCLoRA(),
        )

        # ----------------- Define LoRA modules for q/k/v and out -----------------
        self.attn1.to_q_TC_lora = Linear_TCLoRA(
            in_features=self.attn1.to_q.in_features,
            out_features=self.attn1.to_q.out_features,
            r=TC_lora_rank,
            alpha=TC_lora_alpha,
        )

        self.attn1.to_k_TC_lora = Linear_TCLoRA(
            in_features=self.attn1.to_k.in_features,
            out_features=self.attn1.to_k.out_features,
            r=TC_lora_rank,
            alpha=TC_lora_alpha,
        )

        self.attn1.to_v_TC_lora = Linear_TCLoRA(
            in_features=self.attn1.to_v.in_features,
            out_features=self.attn1.to_v.out_features,
            r=TC_lora_rank,
            alpha=TC_lora_alpha,
        )

        self.attn1.to_out_TC_lora = Linear_TCLoRA(
            in_features=self.attn1.to_out[0].in_features,
            out_features=self.attn1.to_out[0].out_features,
            r=TC_lora_rank,
            alpha=TC_lora_alpha,
        )
        
        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward_TCLoRA(
            dim,
            dropout=dropout,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
            r=TC_lora_rank,
            alpha=TC_lora_alpha,          
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        res_hidden_states: torch.Tensor, 
        res_ff0_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            res_hidden_states=res_hidden_states, 
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states, res_ff0_hidden_states, res_hidden_states, text_seq_length)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states