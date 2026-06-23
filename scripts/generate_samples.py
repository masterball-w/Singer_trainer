"""用微调后的 MusicGen 生成测试音乐。"""

import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from transformers import MusicgenForConditionalGeneration, AutoProcessor

FINETUNED_DIR = Path("checkpoints/finetuned/final")
OUTPUT_DIR = Path("generated_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 生成配置
PROMPTS = [
    "a beautiful piano melody",
    "upbeat electronic dance music",
    "calm ambient music with soft pads",
]
DURATION = 10  # 秒
SAMPLE_RATE = 32000


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading fine-tuned model...")
    processor = AutoProcessor.from_pretrained(str(FINETUNED_DIR))
    model = MusicgenForConditionalGeneration.from_pretrained(
        str(FINETUNED_DIR),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    model.to(device)
    model.eval()

    print(f"Model loaded from {FINETUNED_DIR}")
    print(f"Generating {len(PROMPTS)} samples, {DURATION}s each...\n")

    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] Generating: '{prompt}'...")

        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=int(DURATION * 50),  # ~50 tokens/sec for EnCodec
                do_sample=True,
                temperature=1.0,
                top_k=250,
                guidance_scale=3.0,
            )

        # audio_values shape: [B, C, T] or [B, T]
        audio = audio_values[0].cpu().numpy().astype(np.float32)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=0)  # 混合为单声道

        # 归一化
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.9

        out_path = OUTPUT_DIR / f"sample_{i+1}_{prompt[:20].replace(' ', '_')}.wav"
        sf.write(str(out_path), audio, SAMPLE_RATE)
        print(f"  Saved: {out_path}")

    print(f"\nAll samples saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
