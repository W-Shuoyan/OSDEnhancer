from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class CrossAttention2D(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.kv_proj = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, groups=dim * 2, bias=True)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        nn.init.zeros_(self.project_out.weight)

    def forward(self, x, y):
        B, C, H, W = x.shape
        head_dim = C // self.num_heads

        q = self.q_dwconv(self.q_proj(x))
        kv = self.kv_dwconv(self.kv_proj(y))
        k, v = torch.chunk(kv, 2, dim=1)

        q = q.view(B, self.num_heads, head_dim, H * W)
        k = k.view(B, self.num_heads, head_dim, H * W)
        v = v.view(B, self.num_heads, head_dim, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)    # [B, head, c, HW]

        out = out.view(B, C, H, W)
        out = self.project_out(out)
        out = out + x 
        return out    

class DCNv2Pack(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        deformable_groups=1,  # mapped to DeformConv2d `groups`
        bias=True,
    ):
        super().__init__()
        assert in_channels % deformable_groups == 0, \
            f"in_channels({in_channels}) must be divisible by deformable_groups({deformable_groups})"
        assert out_channels % deformable_groups == 0, \
            f"out_channels({out_channels}) must be divisible by deformable_groups({deformable_groups})"

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = deformable_groups  # torchvision ties conv groups to deformable groups

        kk = kernel_size * kernel_size
        # offset: 2 * k * k * groups; mask: 1 * k * k * groups
        self.conv_offset = nn.Conv2d(in_channels, 2 * self.groups * kk, kernel_size=3, stride=1, padding=1)
        self.conv_mask   = nn.Conv2d(in_channels, 1 * self.groups * kk, kernel_size=3, stride=1, padding=1)

        self.deform = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=self.groups,
            bias=bias,
        )

        self._init_parameters()

    def _init_parameters(self):
        # Start near standard conv: offset = 0, mask ≈ 0.5
        nn.init.constant_(self.conv_offset.weight, 0.0)
        nn.init.constant_(self.conv_offset.bias, 0.0)
        nn.init.constant_(self.conv_mask.weight, 0.0)
        nn.init.constant_(self.conv_mask.bias, 0.0)

        nn.init.kaiming_normal_(self.deform.weight, nonlinearity="leaky_relu")
        if self.deform.bias is not None:
            nn.init.constant_(self.deform.bias, 0.0)

    def forward(self, x, offset_feat):
        orig_dtype = x.dtype
        with torch.amp.autocast(device_type="cuda", enabled=False):
            x_f32 = x.float()
            offset_feat_f32 = offset_feat.float()
            offset = self.conv_offset(offset_feat_f32)
            mask = torch.sigmoid(self.conv_mask(offset_feat_f32))
            out_f32 = self.deform(x_f32, offset, mask)
        return out_f32.to(orig_dtype)
    

class PCDAlignment(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8, has_lower_offset=False):
        super().__init__()
        self.has_lower_offset = has_lower_offset
        self.offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if self.has_lower_offset:
            self.offset_conv12 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        # Single deformable alignment (groups = deformable_groups)
        self.dcn_pack = DCNv2Pack(
            in_channels=num_feat,
            out_channels=num_feat,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=deformable_groups,
            bias=True,
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, cur_feat, pre_feat, lower_offset=None):
        offset_feat = torch.cat([cur_feat, pre_feat], dim=1)  # [B, 2C, H, W]
        offset_feat = self.lrelu(self.offset_conv1(offset_feat))  # [B, C, H, W]
        if self.has_lower_offset and lower_offset is not None:
            offset_feat = self.lrelu(self.offset_conv12(torch.cat([offset_feat, lower_offset * 2], dim=1)))
        offset_feat = self.lrelu(self.offset_conv2(offset_feat))  # [B, C, H, W]
        feat = self.dcn_pack(pre_feat, offset_feat)
        feat = self.lrelu(feat)
        return feat, offset_feat


class StrictCogVideoXUpsample3D(nn.Module):
    r"""
    3D upsample layer for video tensors [B, C, T, H, W].

    Args:
        in_channels (`int`):
            Number of channels in the input.
        out_channels (`int`):
            Number of channels produced by the convolution.
        kernel_size (`int`, defaults to `3`):
            Size of the convolving kernel.
        stride (`int`, defaults to `1`):
            Stride of the convolution.
        padding (`int`, defaults to `1`):
            Padding size.
        spatial_scale (`int`, defaults to `2`):
            Spatial upsample factor (H, W). 1 means no spatial upsample.
        temporal_scale (`int`, defaults to `1`):
            Temporal upsample factor (T). 1 means no temporal upsample.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        spatial_scale: int = 2,
        temporal_scale: int = 1,
    ) -> None:
        super().__init__()

        assert spatial_scale in (1, 2, 4), "spatial_scale must be in {1, 2, 4}"
        assert temporal_scale in (1, 2, 4), "temporal_scale must be in {1, 2, 4}"

        # Per-frame 2D convolution after upsampling
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Store scales as floats for interpolate
        self.spatial_scale = float(spatial_scale)
        self.temporal_scale = float(temporal_scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: [B, C, T, H, W]
        Output T = T * temporal_scale, H/W = H/W * spatial_scale (if > 1).
        """
        b, c, t, h, w = inputs.shape

        # 1) joint temporal + spatial upsampling when temporal_scale > 1
        if self.temporal_scale > 1.0:
            scale_t = self.temporal_scale
            scale_s = self.spatial_scale

            # Interpolate over [T, H, W] directly
            if scale_s != 1.0:
                scale_factor = (scale_t, scale_s, scale_s)
            else:
                scale_factor = (scale_t, 1.0, 1.0)

            inputs = F.interpolate(
                inputs,
                scale_factor=scale_factor,
            )

        # 2) spatial-only upsampling when temporal_scale == 1
        elif self.spatial_scale != 1.0:
            # Treat (B, T) as batch dimension for 2D interpolation
            inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            inputs = F.interpolate(
                inputs,
                scale_factor=self.spatial_scale,
            )
            _, c2, h2, w2 = inputs.shape
            inputs = inputs.reshape(b, t, c2, h2, w2).permute(0, 2, 1, 3, 4)

        # 3) per-frame 2D convolution
        b, c, t, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        inputs = self.conv(inputs)
        inputs = inputs.reshape(b, t, *inputs.shape[1:]).permute(0, 2, 1, 3, 4)

        return inputs

    
class DecodeBlock_BD(nn.Module):
    def __init__(
        self, 
        in_ch: int,
        has_lower_offset: bool = False, 
        lower_ch: Optional[int] = None,
        dec_ver_spatial_scale: Optional[int] = None,
        dec_ver_temporal_scale: Optional[int] = None,
    ):
        super().__init__()

        # Temporal interactions (forward & backward)
        if has_lower_offset:
            self.ver_align = StrictCogVideoXUpsample3D(lower_ch, in_ch, spatial_scale=dec_ver_spatial_scale, temporal_scale=dec_ver_temporal_scale)
        else:
            self.ver_align = None
        
        self.align = PCDAlignment(in_ch, has_lower_offset=has_lower_offset)

        self.attn = CrossAttention2D(in_ch)

    def fake_context_parallel_forward(
        self, inputs: torch.Tensor, conv_cache: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if conv_cache is not None:
            inputs = torch.cat([conv_cache] + [inputs], dim=2)
        return inputs
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        conv_cache: Optional[Dict[str, torch.Tensor]] = None,
        lower_offsets: Optional[torch.Tensor] = None, 
        is_reverse: Optional[torch.Tensor] = False,
    ):
        new_conv_cache = {}
        conv_cache = conv_cache or {}
        hidden_states = self.fake_context_parallel_forward(hidden_states, conv_cache.get('align'))
        hidden_states = list(hidden_states.unbind(dim=2))
        T = len(hidden_states)

        if self.ver_align is not None and lower_offsets is not None:
            lower_offsets = self.ver_align(lower_offsets)
            lower_offsets = list(lower_offsets.unbind(dim=2))
        else:
            lower_offsets = [None] * (T - 1)
        if is_reverse:
            for t in reversed(range(T)):
                if t < T - 1:
                    h_next, lower_offsets[t] = self.align(hidden_states[t], h_next, lower_offsets[t])
                    hidden_states[t] = self.attn(hidden_states[t], h_next)
                    h_next = hidden_states[t]
                else:
                    h_next = hidden_states[t]
        else:
            for t in range(T):
                if t > 0:
                    h_pre, lower_offsets[t-1] = self.align(hidden_states[t], h_pre, lower_offsets[t-1])
                    hidden_states[t] = self.attn(hidden_states[t], h_pre)
                    h_pre = hidden_states[t]
                else:
                    h_pre = hidden_states[t]

        if conv_cache is not None and conv_cache.get("align") is not None:
            hidden_states = hidden_states[1:]
        hidden_states = torch.stack(hidden_states, dim=2)
        lower_offsets = torch.stack(lower_offsets, dim=2)
        new_conv_cache['align'] = hidden_states[:, :, -1:].clone()

        return hidden_states, new_conv_cache, lower_offsets