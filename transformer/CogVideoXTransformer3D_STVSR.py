from typing import Any, Dict, Optional, Tuple, Union, List
from types import MethodType
import os, glob

import torch
from torch import nn

from diffusers import CogVideoXTransformer3DModel
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.model_loading_utils import load_state_dict

from peft import LoraConfig, get_peft_model

from .modules import CogVideoXPatchResEmbed, CogVideoX_TCLoRA_Block

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class CogVideoXTransformer3D_STVSR_Model(CogVideoXTransformer3DModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        res_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any TCs and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        res_hidden_states, res_ff0_hidden_states = self.res_patch(res_hidden_states)

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    res_hidden_states, 
                    res_ff0_hidden_states,
                    emb,
                    image_rotary_emb,
                    attention_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    res_hidden_states=res_hidden_states, 
                    res_ff0_hidden_states=res_ff0_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                )

        hidden_states = self.norm_final(hidden_states)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
    
    @classmethod
    def from_pretrained(
        cls,
        ckpt_path,
        **kwargs,
    ):
        config, unused_kwargs = CogVideoXTransformer3DModel.load_config(
            ckpt_path,
            return_unused_kwargs=True,
            **kwargs,
        )
        model = cls.from_config(config, **unused_kwargs)
        cfg = model.config
        new_blocks = nn.ModuleList()
        for _, _old_block in enumerate(model.transformer_blocks):
            TC_block = CogVideoX_TCLoRA_Block(
                dim=cfg.num_attention_heads * cfg.attention_head_dim,
                num_attention_heads=cfg.num_attention_heads,
                attention_head_dim=cfg.attention_head_dim,
                time_embed_dim=cfg.time_embed_dim,
                dropout=cfg.dropout,
                activation_fn=cfg.activation_fn,
                attention_bias=cfg.attention_bias,
                norm_elementwise_affine=cfg.norm_elementwise_affine,
                norm_eps=cfg.norm_eps,
                TC_lora_rank=128,
                TC_lora_alpha=128,
            )
            new_blocks.append(TC_block)

        model.transformer_blocks = new_blocks

        model.res_patch = CogVideoXPatchResEmbed(
            patch_size=cfg.patch_size,
            patch_size_t=cfg.patch_size_t,
            in_channels=cfg.in_channels,
            dim_head=cfg.attention_head_dim,
            heads=cfg.num_attention_heads,
        )
        pattern = os.path.join(
            ckpt_path,
            "diffusion_pytorch_model*.safetensors",
        )
        TC_files = sorted(glob.glob(pattern))
        merged_state_dict = {}
        for wf in TC_files:
            shard = load_state_dict(wf)
            merged_state_dict.update(shard)
        TE_lora_opt = LoraConfig(
            r=128,
            lora_alpha=128,
            init_lora_weights=False,
            target_modules=["to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2", "proj_out"],
            task_type="FEATURE_EXTRACTION",
        )
        peft_model = get_peft_model(model, TE_lora_opt, adapter_name="TE_lora")
        peft_model.merge_adapter()
        model = peft_model.model
        model.load_state_dict(merged_state_dict, strict=False)
        return model