import torch 
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from huggingface_hub import snapshot_download
from typing import Dict, Tuple, Union, Optional
from tqdm import tqdm

from diffusers.pipelines.pipeline_utils import DiffusionPipeline 
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.utils import logging
from safetensors.torch import load_file

from vae.AutoencoderKLCogVideoX_STVSR import AutoencoderKLCogVideoX_STVSR
from vae.modules import DCNv2Pack
from transformer.CogVideoXTransformer3D_STVSR import CogVideoXTransformer3D_STVSR_Model

logger = logging.get_logger(__name__)

    
class OSDEnhancerPipeline(DiffusionPipeline): 
    def __init__(
        self,
        vae: AutoencoderKLCogVideoX,
        transformer: nn.Module,
        scheduler: CogVideoXDPMScheduler,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.prompt_embedding = None
        self.model_cpu_offload_seq = "transformer->vae"

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        local_files_only: bool = False,
    ):
        device = torch.device(device if torch.cuda.is_available() else "cpu")

        print("[Load] OSDEnhancerPipeline loading...")
        print(f"[Load] checkpoint: {pretrained_model_name_or_path}")
        print(f"[Load] device={device}, dtype={torch_dtype}")

        input_path = Path(pretrained_model_name_or_path)

        if input_path.is_dir():
            ckpt_path = input_path
            print(f"[Load] Using local checkpoint: {ckpt_path}")
        else:
            print("[Load] Downloading checkpoint from Hugging Face Hub...")
            ckpt_path = Path(
                snapshot_download(
                    repo_id=pretrained_model_name_or_path,
                    local_files_only=local_files_only,
                    allow_patterns=[
                        "scheduler/*",
                        "vae/*",
                        "transformer/*",
                        "prompt_embeddings/*",
                    ],
                )
            )
            print(f"[Load] Checkpoint cached at: {ckpt_path}")

        scheduler = CogVideoXDPMScheduler.from_pretrained(
            ckpt_path,
            subfolder="scheduler",
            local_files_only=True,
        )
        print("[Load] Scheduler loading finished.")

        vae = AutoencoderKLCogVideoX_STVSR.from_pretrained(
            ckpt_path,
            subfolder="vae",
            torch_dtype=torch_dtype,
            local_files_only=True,
        )

        for p in vae.parameters():
            p.requires_grad = False

        vae.to(device=device)

        for m in vae.modules():
            if isinstance(m, DCNv2Pack):
                m.to(dtype=torch.float32)

        vae.eval()
        print("[Load] VAE loading finished.")

        transformer = CogVideoXTransformer3D_STVSR_Model.from_pretrained(
            ckpt_path=ckpt_path / "transformer"
        )

        for p in transformer.parameters():
            p.requires_grad = False

        transformer.to(device=device, dtype=torch_dtype)
        transformer.eval()
        print("[Load] Transformer loading finished.")

        prompt_embedding_path = ckpt_path / "prompt_embeddings" / "empty.safetensors"

        prompt_embedding = load_file(str(prompt_embedding_path))["prompt_embedding"]
        prompt_embedding = prompt_embedding.to(
            device=device,
            dtype=torch_dtype,
        ).unsqueeze(0).contiguous()

        print("[Load] Prompt embedding loading finished.")

        pipe = cls(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        pipe.prompt_embedding = prompt_embedding
        pipe.to(device)

        print("[Load] OSDEnhancerPipeline loading finished.")

        return pipe
    
    def get_resize_crop_region_for_grid(self, src, tgt_width, tgt_height):
        tw = tgt_width
        th = tgt_height
        h, w = src
        r = h / w
        if r > (th / tw):
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        crop_top = int(round((th - resize_height) / 2.0))
        crop_left = int(round((tw - resize_width) / 2.0))

        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        p = transformer_config.patch_size
        p_t = transformer_config.patch_size_t

        base_size_width = transformer_config.sample_width // p
        base_size_height = transformer_config.sample_height // p

        if p_t is None:
            # CogVideoX 1.0
            grid_crops_coords = self.get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=transformer_config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=transformer_config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(grid_height, grid_width),
                device=device,
            )

        return freqs_cos, freqs_sin

    def spatial_upsampling(self, x: torch.Tensor, scale: Union[int, float]) -> torch.Tensor:
        if scale == 1:
            return x
        B, M, C, H, W = x.shape
        x_ = x.reshape(B * M, C, H, W)
        y_ = F.interpolate(
            x_,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False
        )
        H_out, W_out = y_.shape[-2:]
        return y_.reshape(B, M, C, H_out, W_out)
    
    def temporal_blending(self, frames: torch.Tensor, k: int) -> torch.Tensor:
        assert frames.ndim == 5, "frames must be (b, N, 3, h, w)"
        b, N, c, h, w = frames.shape
        assert c == 3, "frames must be RGB"
        assert k >= 0, "the number of interpolated frames k must be >= 0"
        if k == 0:
            return frames
        t_list = [i / (k + 1) for i in range(1, k + 1)]
        seq_outputs = []
        for idx in range(N - 1):
            im0 = frames[:, idx]     # (b, 3, h, w)
            im1 = frames[:, idx + 1] # (b, 3, h, w)
            if idx == 0:
                seq_outputs.append(im0.unsqueeze(1))  # (b,1,3,h,w)
            # I_t = (1 - t) * I0 + t * I1
            for t in t_list:
                middle = (1.0 - t) * im0 + t * im1    # (b,3,h,w)
                seq_outputs.append(middle.unsqueeze(1))
            seq_outputs.append(im1.unsqueeze(1))
        seq = torch.cat(seq_outputs, dim=1)  # (b, M, 3, h, w)
        return seq

    def residuals_buliding(self, frames: torch.Tensor, k: int) -> torch.Tensor:
        b, N, c, h, w = frames.shape
        t_list = [i / (k + 1) for i in range(1, k + 1)]  # only used for count
        residual_outputs = []
        for idx in range(N - 1):
            I0 = frames[:, idx]       # (b, 3, h, w)
            I1 = frames[:, idx + 1]   # (b, 3, h, w)
            # First reference frame: residual = 0
            if idx == 0:
                res_ref0 = torch.zeros_like(I0)
                residual_outputs.append(res_ref0.unsqueeze(1))  # (b, 1, 3, h, w)
            # Intermediate frames: copy residual I1 - I0
            res_pair = I1 - I0  # (b, 3, h, w)
            for _ in t_list:
                residual_outputs.append(res_pair.unsqueeze(1))
            # Right reference frame of this pair: residual = 0
            res_ref1 = torch.zeros_like(I1)
            residual_outputs.append(res_ref1.unsqueeze(1))
        return torch.cat(residual_outputs, dim=1)  # (b, T, 3, h, w)

    def pad_to_multiple(self, x, multiple=16, mode="reflect"):
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B, C, T, H, W], but got shape: {x.shape}")
        B, C, T, H, W = x.shape
        # If T already 8N+1, pad_t = 0
        # Solve for smallest T_target >= T such that T_target = 8N + 1
        if T <= 1:
            T_target = 1
        else:
            T_target = ((T - 1 + 7) // 8) * 8 + 1
        pad_t = T_target - T
        if pad_t > 0:
            last_frame = x[:, :, -1:].expand(B, C, pad_t, H, W)
            x = torch.cat([x, last_frame], dim=2)
            T = T_target
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h == 0 and pad_w == 0 and pad_t == 0:
            return x, (0, 0, 0, 0, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(
            x,
            (pad_left, pad_right, pad_top, pad_bottom, 0, 0),
            mode=mode,
        )
        pads = (pad_t, pad_left, pad_right, pad_top, pad_bottom)
        return x, pads

    def unpad_from_multiple(self, x, pads):
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B, C, T, H, W], but got shape: {x.shape}")
        pad_t, pad_left, pad_right, pad_top, pad_bottom = pads
        B, C, T, H, W = x.shape
        if pad_t > 0:
            x = x[:, :, : T - pad_t, ...]
        h_start = pad_top
        h_end = H - pad_bottom if pad_bottom > 0 else H
        w_start = pad_left
        w_end = W - pad_right if pad_right > 0 else W
        x = x[:, :, :, h_start:h_end, w_start:w_end]
        return x

    @torch.no_grad()
    def __call__(
        self,
        input: torch.Tensor,
        spatial_scale: Union[int, float] = 4,
        temporal_scale: int = 2,
        chunk_length: Optional[int] = None,
        overlap: Optional[int] = None,
    ):
        base_transformer = getattr(self.transformer, "module", self.transformer)

        base_vae = getattr(self.vae, "module", self.vae)
        base_vae.enable_slicing()
        base_vae.enable_tiling()

        dtype = next(base_transformer.parameters()).dtype
        device = next(base_transformer.parameters()).device

        vae_config = base_vae.config
        transformer_config = base_transformer.config

        B_in, T_in, C_in, H_in, W_in = input.shape
        print(f"[Input]  frames={T_in}, height={H_in}, width={W_in}.")

        init = self.spatial_upsampling(
            self.temporal_blending(input, temporal_scale - 1),
            spatial_scale
        ).clamp(0, 1)

        res = self.residuals_buliding(
            self.spatial_upsampling(input, spatial_scale).clamp(0, 1),
            temporal_scale - 1
        )

        B, T_all, C, H, W = init.shape
        print(f"[Output Target] frames={T_all}, height={H}, width={W}.")

        outputs = []
        starts = [0]
        prev_ed = None

        use_chunk = chunk_length is not None

        if use_chunk:
            assert chunk_length >= 1, "chunk_length must be >= 1"
            assert (chunk_length - 1) % 8 == 0, "chunk_length must satisfy 8N+1, e.g., 9, 17, 25, ..."
            
            overlap = 1 if overlap is None else overlap
            assert overlap >= 0, "overlap must be >= 0"
            assert overlap < chunk_length, "overlap must be smaller than chunk_length"

            base_len = min(chunk_length, T_all)

            starts = [0]

            if base_len < T_all:
                step_t = base_len - overlap

                while True:
                    ns = starts[-1] + step_t
                    if ns + base_len >= T_all:
                        break
                    starts.append(ns)

                last_start = T_all - base_len
                if last_start > starts[-1]:
                    starts.append(last_start)

            print(
                f"[Chunk] enabled=True, "
                f"chunk_length={base_len}, "
                f"num_chunks={len(starts)}, "
                f"overlap={overlap}"
            )
        else:
            base_len = T_all

        iterator = enumerate(starts)
        if use_chunk:
            iterator = tqdm(
                iterator,
                total=len(starts),
                desc="[Chunk Progress]",
                ncols=100
            )

        for ci, st in iterator:
            ed = min(st + base_len, T_all)

            init_chunk = init[:, st:ed, ...]
            res_chunk = res[:, st:ed, ...]

            init_chunk = (init_chunk * 2 - 1).to(device, dtype=base_vae.dtype)
            init_chunk = init_chunk.permute(0, 2, 1, 3, 4).contiguous()
            init_chunk, pads = self.pad_to_multiple(init_chunk)

            res_chunk = res_chunk.to(device, dtype=base_vae.dtype)
            res_chunk = res_chunk.permute(0, 2, 1, 3, 4).contiguous()
            res_chunk, _ = self.pad_to_multiple(res_chunk)

            timesteps = torch.full(
                (init_chunk.shape[0],),
                399,
                device=device,
                dtype=torch.long,
            )

            print("[Status] VAE encoding...")

            init_latent = base_vae.encode(init_chunk).latent_dist.sample()
            init_latent = init_latent * vae_config.scaling_factor
            init_latent = init_latent.to(dtype=dtype).permute(0, 2, 1, 3, 4).contiguous()

            res_latent = base_vae.encode(res_chunk).latent_dist.sample()
            res_latent = res_latent * vae_config.scaling_factor
            res_latent = res_latent.to(dtype=dtype).permute(0, 2, 1, 3, 4).contiguous()

            del init_chunk, res_chunk
            torch.cuda.empty_cache()

            patch_size_t = transformer_config.patch_size_t
            ncopy = 0

            if patch_size_t is not None:
                ncopy = (patch_size_t - (init_latent.shape[1] % patch_size_t)) % patch_size_t

                if ncopy > 0:
                    init_latent = torch.cat(
                        [
                            init_latent[:, :1].repeat(1, ncopy, 1, 1, 1),
                            init_latent
                        ],
                        dim=1
                    )
                    res_latent = torch.cat(
                        [
                            res_latent[:, :1].repeat(1, ncopy, 1, 1, 1),
                            res_latent
                        ],
                        dim=1
                    )

            vae_scale_factor_spatial = 2 ** (len(vae_config.block_out_channels) - 1)

            rotary_emb = (
                self.prepare_rotary_positional_embeddings(
                    height=init_latent.shape[3] * vae_scale_factor_spatial,
                    width=init_latent.shape[4] * vae_scale_factor_spatial,
                    num_frames=init_latent.shape[1],
                    transformer_config=transformer_config,
                    vae_scale_factor_spatial=vae_scale_factor_spatial,
                    device=device,
                )
                if transformer_config.use_rotary_positional_embeddings
                else None
            )

            print("[Status] DiT inference...")

            predicted_noise = base_transformer(
                hidden_states=init_latent,
                encoder_hidden_states=self.prompt_embedding,
                res_hidden_states=res_latent,
                timestep=timesteps,
                image_rotary_emb=rotary_emb,
                return_dict=False,
            )[0]

            latent_pred = self.scheduler.get_velocity(
                predicted_noise,
                init_latent,
                timesteps
            )

            del predicted_noise, init_latent, res_latent
            torch.cuda.empty_cache()

            if patch_size_t is not None and ncopy > 0:
                latent_pred = latent_pred[:, ncopy:]

            latent_pred = latent_pred.to(dtype=base_vae.dtype)
            latent_pred = latent_pred.permute(0, 2, 1, 3, 4).contiguous()
            latent_pred = latent_pred / vae_config.scaling_factor

            print("[Status] VAE decoding...")

            out = base_vae.decode(latent_pred).sample.to(torch.float32)

            out = self.unpad_from_multiple(out, pads)
            out = out.permute(0, 2, 1, 3, 4).contiguous()
            out = (out * 0.5 + 0.5).clamp(0, 1)

            if prev_ed is not None:
                drop = max(0, prev_ed - st)
                if drop > 0:
                    out = out[:, drop:, ...]

            outputs.append(out.cpu())
            prev_ed = ed

            del latent_pred, out
            torch.cuda.empty_cache()

        outputs = torch.cat(outputs, dim=1)

        B_out, T_out, C_out, H_out, W_out = outputs.shape
        print(f"[Output] frames={T_out}, height={H_out}, width={W_out}.")

        return outputs