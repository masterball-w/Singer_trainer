"""Generate a Genshin Impact themed song with AudioX-Turbo.

Generates multiple 10-second segments with carefully crafted prompts
emphasizing vocals, then concatenates with crossfade into a full song.
"""

import os
import sys
import time
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from einops import rearrange

# HF mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(Path("checkpoints/huggingface").resolve())
os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path("checkpoints/huggingface").resolve())
os.environ["TRANSFORMERS_CACHE"] = str(Path("checkpoints/huggingface").resolve())

from audiox_turbo.inference import load_audiox_turbo_model
from audiox_turbo.inference.generation import generate_diffusion_cond_dmd

CONFIG_PATH = "configs/audiox_turbo_infer_4step.json"
CKPT_PATH = "checkpoints/audiox_turbo/audiox_turbo.ckpt"
VAE_PATH = "checkpoints/pretransform/vae.ckpt"
OUTPUT_DIR = Path("generated_song_genshin")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DURATION = 10  # seconds per segment

# Song structure: each segment has a specific musical role
# Prompts are in English (T5 was trained primarily on English)
# Emphasize vocals/singing in every prompt
SEGMENTS = [
    {
        "name": "intro",
        "prompt": (
            "Epic cinematic orchestral intro with ethereal female humming, "
            "shimmering strings, gentle piano arpeggios, rising tension, "
            "fantasy adventure atmosphere, reverb, 90 BPM, key of A minor"
        ),
    },
    {
        "name": "verse1",
        "prompt": (
            "Beautiful female vocal singing a soft melancholic melody with piano accompaniment, "
            "gentle orchestral strings, light percussion, emotional ballad style, "
            "Japanese anime soundtrack feeling, 90 BPM, A minor, clear vocals with reverb"
        ),
    },
    {
        "name": "verse2",
        "prompt": (
            "Female vocalist singing an emotional verse with acoustic guitar and strings, "
            "building intensity, soft drums entering, cinematic orchestral pop, "
            "epic fantasy soundtrack with clear singing voice, 90 BPM"
        ),
    },
    {
        "name": "chorus",
        "prompt": (
            "Powerful female vocal chorus, soaring melody, full orchestra with dramatic timpani, "
            "epic cinematic climax, layered harmonies, majestic strings and brass, "
            "anime opening style, emotional and uplifting, 90 BPM, A minor to C major"
        ),
    },
    {
        "name": "verse3",
        "prompt": (
            "Female singer continuing with emotional verse, piano and electric guitar, "
            "atmospheric pads, layered vocal harmonies, building towards final chorus, "
            "cinematic orchestral pop ballad, 90 BPM"
        ),
    },
    {
        "name": "chorus_final",
        "prompt": (
            "Grand epic final chorus with powerful female vocals and choir, "
            "full orchestra crescendo, dramatic brass, thundering timpani, "
            "triumphant and emotional climax, anime soundtrack finale, 90 BPM"
        ),
    },
    {
        "name": "outro",
        "prompt": (
            "Gentle orchestral outro with soft female humming fading away, "
            "delicate piano melody, ethereal pads, peaceful resolution, "
            "cinematic ending, diminuendo, reverb tail, 90 BPM, A minor"
        ),
    },
]


def crossfade(a, b, fade_len):
    """Crossfade between two audio arrays (numpy)."""
    # a: (channels, samples), b: (channels, samples)
    fade_a = np.linspace(1, 0, fade_len).reshape(1, -1)
    fade_b = np.linspace(0, 1, fade_len).reshape(1, -1)

    a_tail = a[:, -fade_len:] * fade_a
    b_head = b[:, :fade_len] * fade_b
    crossfaded = a_tail + b_head

    result = np.concatenate([
        a[:, :-fade_len],
        crossfaded,
        b[:, fade_len:],
    ], axis=1)
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram:.1f} GB")

    print("\nLoading AudioX-Turbo model...")
    t0 = time.time()
    model, model_config = load_audiox_turbo_model(
        CONFIG_PATH, CKPT_PATH,
        pretransform_ckpt_path=VAE_PATH, device=device,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    sample_rate = model_config["sample_rate"]
    target_fps = model_config.get("video_fps", 5)
    sample_size = model_config["sample_size"]

    all_segments = []
    total = len(SEGMENTS)

    for i, seg in enumerate(SEGMENTS):
        print(f"\n[{i+1}/{total}] Generating '{seg['name']}'...")
        print(f"  Prompt: {seg['prompt'][:60]}...")

        video_tensor = torch.zeros(DURATION * target_fps, 3, 224, 224)
        sync_features = torch.zeros(1, 240, 768, device=device)
        audio_tensor = torch.zeros(2, sample_rate * DURATION)

        conditioning = [{
            "video_prompt": {
                "video_tensors": video_tensor.unsqueeze(0),
                "video_sync_frames": sync_features,
            },
            "text_prompt": seg["prompt"],
            "audio_prompt": audio_tensor.unsqueeze(0),
            "seconds_start": 0,
            "seconds_total": DURATION,
        }]

        t1 = time.time()
        output = generate_diffusion_cond_dmd(
            model,
            steps=4,
            conditioning=conditioning,
            sample_size=sample_size,
            seed=42 + i,
            device=device,
        )
        gen_time = time.time() - t1

        # Post-process
        output = output[:, :, : sample_rate * DURATION]
        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32)
        max_val = torch.max(torch.abs(output)).clamp_min(1e-8)
        output = output.div(max_val).clamp(-1, 1)

        audio_np = output.cpu().numpy()
        all_segments.append(audio_np)

        # Save individual segment
        seg_path = OUTPUT_DIR / f"seg_{i+1}_{seg['name']}.wav"
        sf.write(str(seg_path), audio_np.T, sample_rate)
        print(f"  Done in {gen_time:.1f}s -> {seg_path.name}")

    # Concatenate with crossfade
    print(f"\nConcatenating {len(all_segments)} segments with crossfade...")
    fade_seconds = 1.5
    fade_samples = int(fade_seconds * sample_rate)

    song = all_segments[0]
    for seg_audio in all_segments[1:]:
        song = crossfade(song, seg_audio, fade_samples)

    # Final normalize
    max_val = np.max(np.abs(song))
    if max_val > 0:
        song = song / max_val * 0.95

    total_duration = song.shape[1] / sample_rate
    song_path = OUTPUT_DIR / "genshin_star_traveler_full.wav"
    sf.write(str(song_path), song.T, sample_rate)
    print(f"\nFull song saved: {song_path}")
    print(f"Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Format: {sample_rate}Hz, stereo, float32")


if __name__ == "__main__":
    main()
