"""AudioX-Turbo text-to-music inference test."""

import os
import sys
import torch
import soundfile as sf
from pathlib import Path
from einops import rearrange

# Set HF mirror for downloading CLIP/T5 models
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(Path("checkpoints/huggingface").resolve())
os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path("checkpoints/huggingface").resolve())
os.environ["TRANSFORMERS_CACHE"] = str(Path("checkpoints/huggingface").resolve())

from audiox_turbo.inference import load_audiox_turbo_model
from audiox_turbo.inference.generation import generate_diffusion_cond_dmd

CONFIG_PATH = "configs/audiox_turbo_infer_4step.json"
CKPT_PATH = "checkpoints/audiox_turbo/audiox_turbo.ckpt"
VAE_PATH = "checkpoints/pretransform/vae.ckpt"
OUTPUT_DIR = Path("generated_samples_audiox")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "A beautiful piano melody with soft harmonies",
    "Upbeat electronic dance music with synthesizers",
    "Calm ambient music with ethereal pads and gentle textures",
]
DURATION = 10  # seconds


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print(f"\nLoading AudioX-Turbo model...")
    model, model_config = load_audiox_turbo_model(
        CONFIG_PATH,
        CKPT_PATH,
        pretransform_ckpt_path=VAE_PATH,
        device=device,
    )
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    target_fps = model_config.get("video_fps", 5)

    print(f"Model loaded. Sample rate: {sample_rate}, Sample size: {sample_size}")
    print(f"Generating {len(PROMPTS)} samples, {DURATION}s each...\n")

    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] Generating: '{prompt}'...")

        # Text-to-Music: no video, no audio reference
        video_tensor = torch.zeros(DURATION * target_fps, 3, 224, 224)
        sync_features = torch.zeros(1, 240, 768, device=device)
        audio_tensor = torch.zeros(2, sample_rate * DURATION)  # empty audio

        conditioning = [{
            "video_prompt": {
                "video_tensors": video_tensor.unsqueeze(0),
                "video_sync_frames": sync_features,
            },
            "text_prompt": prompt,
            "audio_prompt": audio_tensor.unsqueeze(0),
            "seconds_start": 0,
            "seconds_total": DURATION,
        }]

        # 4-step generation
        output = generate_diffusion_cond_dmd(
            model,
            steps=4,
            conditioning=conditioning,
            sample_size=sample_size,
            seed=42 + i,
            device=device,
        )

        # Post-process: trim, rearrange, normalize
        output = output[:, :, : sample_rate * DURATION]
        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(
            torch.max(torch.abs(output)).clamp_min(1e-8)
        ).clamp(-1, 1)

        out_path = OUTPUT_DIR / f"t2m_{i+1}_{prompt[:25].replace(' ', '_')}.wav"
        sf.write(str(out_path), output.cpu().numpy().T, sample_rate)
        print(f"  Saved: {out_path} ({output.shape[1]/sample_rate:.1f}s, stereo)")

    print(f"\nAll samples saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
