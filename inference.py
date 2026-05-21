import argparse
from pathlib import Path

import imageio.v3 as iio
import torch

from pipeline.OSDEnhancer_pipeline import OSDEnhancerPipeline


def read_mp4(path: str) -> tuple[torch.Tensor, float]:
    meta = iio.immeta(path)
    fps = meta.get("fps", 8.0)

    frames = iio.imread(path)  # [T, H, W, C], uint8

    if frames.ndim != 4:
        raise RuntimeError(f"Expected video shape [T, H, W, C], but got {frames.shape}")

    if frames.shape[-1] == 4:
        frames = frames[..., :3]

    video = torch.from_numpy(frames).float() / 255.0
    video = video.permute(0, 3, 1, 2).unsqueeze(0).contiguous()  # [1, T, 3, H, W]

    return video, float(fps)


def save_mp4(video: torch.Tensor, path: str, fps: float):
    video = video.squeeze(0).detach().cpu().clamp(0, 1)  # [T, 3, H, W]
    video = video.permute(0, 2, 3, 1)  # [T, H, W, 3]
    video = (video * 255.0).round().to(torch.uint8).numpy()

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    iio.imwrite(
        path,
        video,
        fps=float(fps),
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=[
            "-crf", "0",
        ]
    )


HF_REPO_ID = "W-Shuoyan/OSDEnhancer"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--spatial_scale", type=float, default=4.0)
    parser.add_argument("--temporal_scale", type=int, default=2)

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Local checkpoint path. If not specified, load from Hugging Face.",
    )

    parser.add_argument("--chunk_num", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lq_video, input_fps = read_mp4(args.input)

    if args.ckpt_path is None:
        ckpt_path = HF_REPO_ID
        local_files_only = False
        print(f"[Load] No local checkpoint specified. Loading from Hugging Face: {ckpt_path}")
    else:
        ckpt_path = args.ckpt_path
        local_files_only = True
        print(f"[Load] Loading local checkpoint from: {ckpt_path}")

    pipe = OSDEnhancerPipeline.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16,
        device=device,
        local_files_only=local_files_only,
    )

    lq_video = lq_video.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = pipe(
            input=lq_video,
            spatial_scale=args.spatial_scale,
            temporal_scale=args.temporal_scale,
            chunk_num=args.chunk_num,
            overlap=args.overlap,
        )

    output_fps = input_fps * args.temporal_scale
    save_mp4(output, args.output, fps=output_fps)

    print(f"[Done] Saved to {args.output}")


if __name__ == "__main__":
    main()