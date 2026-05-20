from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from diffusers import AutoencoderKLCogVideoX
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import CogVideoXDecoder3D
from diffusers.configuration_utils import register_to_config
from .modules import DecodeBlock_BD

class CogVideoXDecoder3D_STVSR(CogVideoXDecoder3D):
    def __init__(
        self, 
        *args, 
        dec_ver_spatial_scales: Tuple[int, ...] = (0, 2, 2, 2),
        dec_ver_temporal_scales: Tuple[int, ...] = (0, 2, 2, 1),
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.up_deformable_blocks = nn.ModuleList([])
        for i in range(len(self.up_blocks)):
            if i == 0:
                deformable_block = DecodeBlock_BD(in_ch=self.up_blocks[i].resnets[0].in_channels, 
                                                has_lower_offset=False,
                                                lower_ch=None,
                                                dec_ver_spatial_scale=dec_ver_spatial_scales[i],
                                                dec_ver_temporal_scale=dec_ver_temporal_scales[i],
                                                )
            else:
                deformable_block = DecodeBlock_BD(in_ch=self.up_blocks[i].resnets[0].in_channels, 
                                                has_lower_offset=True,
                                                lower_ch=self.up_blocks[i-1].resnets[0].in_channels,
                                                dec_ver_spatial_scale=dec_ver_spatial_scales[i],
                                                dec_ver_temporal_scale=dec_ver_temporal_scales[i],
                                                )           
            self.up_deformable_blocks.append(deformable_block)

    def forward(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        r"""The forward method of the `CogVideoXDecoder3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states, new_conv_cache["conv_in"] = self.conv_in(sample, conv_cache=conv_cache.get("conv_in"))

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            # 1. Mid
            hidden_states, new_conv_cache["mid_block"] = self._gradient_checkpointing_func(
                self.mid_block,
                hidden_states,
                temb,
                sample,
                conv_cache.get("mid_block"),
            )

            # 2. Up
            lower_offsets = None
            is_reverse = False
            for i, up_block in enumerate(self.up_blocks):
                conv_cache_key = f"up_block_{i}"
                if conv_cache_key not in new_conv_cache or new_conv_cache[conv_cache_key] is None:
                    new_conv_cache[conv_cache_key] = {}
                is_reverse = not is_reverse
                if lower_offsets is not None:
                    lower_offsets = -lower_offsets
                hidden_states, deform_cache, lower_offsets = self._gradient_checkpointing_func(
                    self.up_deformable_blocks[i],
                    hidden_states,
                    conv_cache.get(conv_cache_key),
                    lower_offsets,
                    is_reverse
                )
                new_conv_cache[conv_cache_key].update(deform_cache)
                hidden_states, up_cache = self._gradient_checkpointing_func(
                    up_block,
                    hidden_states,
                    temb,
                    sample,
                    conv_cache.get(conv_cache_key),
                )
                new_conv_cache[conv_cache_key].update(up_cache)
        else:
            # 1. Mid
            hidden_states, new_conv_cache["mid_block"] = self.mid_block(
                hidden_states, temb, sample, conv_cache=conv_cache.get("mid_block")
            )

            # 2. Up
            lower_offsets = None
            is_reverse = False
            for i, up_block in enumerate(self.up_blocks):
                conv_cache_key = f"up_block_{i}"
                if conv_cache_key not in new_conv_cache or new_conv_cache[conv_cache_key] is None:
                    new_conv_cache[conv_cache_key] = {}
                is_reverse = not is_reverse
                if lower_offsets is not None:
                    lower_offsets = -lower_offsets
                hidden_states, deform_cache, lower_offsets = self.up_deformable_blocks[i](
                    hidden_states,
                    conv_cache.get(conv_cache_key),
                    lower_offsets,
                    is_reverse
                )
                new_conv_cache[conv_cache_key].update(deform_cache)
                hidden_states, up_cache = up_block(
                    hidden_states, temb, sample, conv_cache=conv_cache.get(conv_cache_key)
                )
                new_conv_cache[conv_cache_key].update(up_cache)

        # 3. Post-process
        hidden_states, new_conv_cache["norm_out"] = self.norm_out(
            hidden_states, sample, conv_cache=conv_cache.get("norm_out")
        )
        hidden_states = self.conv_act(hidden_states)
        hidden_states, new_conv_cache["conv_out"] = self.conv_out(hidden_states, conv_cache=conv_cache.get("conv_out"))

        return hidden_states, new_conv_cache    
    

class AutoencoderKLCogVideoX_STVSR(AutoencoderKLCogVideoX):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXResnetBlock3D"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        latent_channels: int = 16,
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 4,
        sample_height: int = 480,
        sample_width: int = 720,
        scaling_factor: float = 1.15258426,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float, ...]] = None,
        latents_std: Optional[Tuple[float, ...]] = None,
        force_upcast: bool = True,
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
        invert_scale_latents: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            latent_channels=latent_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
            sample_height=sample_height,
            sample_width=sample_width,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            latents_mean=latents_mean,
            latents_std=latents_std,
            force_upcast=force_upcast,
            use_quant_conv=use_quant_conv,
            use_post_quant_conv=use_post_quant_conv,
            invert_scale_latents=invert_scale_latents,
        )

        self.decoder = CogVideoXDecoder3D_STVSR(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )