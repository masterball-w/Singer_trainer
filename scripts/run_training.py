"""MusicGen 微调训练脚本。

使用 Hugging Face transformers 库的 Trainer API 对 MusicGen-small
进行微调训练，学习用户数据集中的音乐风格。

适配 transformers 4.55+ / PyTorch 2.x / CUDA 12.x
"""

import os
import sys
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from loguru import logger
from transformers import (
    MusicgenForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)

# ===== 自定义 Trainer =====
# MusicGen 需要显式的 EnCodec audio codes 作为 labels 才能计算 loss
# labels 形状: [B, T, num_codebooks]，模型内部自动 shift 生成 decoder_input_ids
class MusicGenTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        input_values = inputs["input_values"]

        # 确保音频数据类型与模型一致（fp16）
        model_dtype = next(model.parameters()).dtype
        input_values = input_values.to(dtype=model_dtype)

        # 用 EnCodec 编码音频为离散 codes: [B, frames, codebooks, time]
        with torch.no_grad():
            enc_out = model.audio_encoder.encode(input_values)
            audio_codes = enc_out.audio_codes  # [B, 1, num_codebooks, T]

        # 去掉 frame 维度，转置为 [B, T, num_codebooks] 作为 labels
        B, num_frames, num_codebooks, T = audio_codes.shape
        labels = audio_codes.squeeze(1).permute(0, 2, 1).contiguous()  # [B, T, num_codebooks]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_values=input_values,
            labels=labels,
            return_loss=True,
        )

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
SLICES_DIR = Path("music_dataset/slices")
CHECKPOINT_DIR = Path("checkpoints/pretrained/musicgen-small-hf")
OUTPUT_DIR = Path("checkpoints/finetuned")
LOG_DIR = Path("logs")
SAMPLE_RATE = 32000
MAX_DURATION = 30  # 秒

# 训练超参数
BATCH_SIZE = 1  # MusicGen 显存占用大，使用 batch_size=1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0
SAVE_STEPS = 50
EVAL_STEPS = 50
LOGGING_STEPS = 10
SEED = 42

# 设置随机种子
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class MusicSliceDataset(Dataset):
    """音乐切片数据集。

    加载预处理后的 30 秒 WAV 切片，返回音频波形。
    """

    def __init__(self, slices_dir: Path, max_duration: int = 30, sample_rate: int = 32000):
        self.slices_dir = slices_dir
        self.max_samples = max_duration * sample_rate
        self.sample_rate = sample_rate

        # 收集所有 WAV 切片
        self.files = sorted(slices_dir.glob("*.wav"))
        if not self.files:
            raise RuntimeError(f"没有找到训练数据: {slices_dir}")

        logger.info(f"数据集: {len(self.files)} 个切片 ({slices_dir})")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_path = self.files[idx]
        audio, sr = sf.read(str(wav_path), dtype="float32")

        # 确保单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # 截断或填充
        if len(audio) > self.max_samples:
            audio = audio[: self.max_samples]
        elif len(audio) < self.max_samples:
            padding = np.zeros(self.max_samples - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, padding])

        return {
            "input_values": torch.from_numpy(audio),
            "file_name": wav_path.stem,
        }


def main():
    # ===== 环境检查 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ===== 加载模型和处理器 =====
    logger.info("Loading model and processor...")

    processor = AutoProcessor.from_pretrained(str(CHECKPOINT_DIR))
    model = MusicgenForConditionalGeneration.from_pretrained(
        str(CHECKPOINT_DIR),
        torch_dtype=torch.float32,  # float32 加载，由 Trainer fp16 autocast 处理混合精度
    )

    # 启用 gradient checkpointing 节省显存
    model.gradient_checkpointing_enable()
    logger.info("Enabled gradient checkpointing for memory efficiency")

    # 冻结 audio_encoder (EnCodec) — 只微调 Transformer 部分
    if hasattr(model, "audio_encoder"):
        for param in model.audio_encoder.parameters():
            param.requires_grad = False
        logger.info("Frozen: audio_encoder (EnCodec)")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Parameters: {trainable_params:,} trainable / {total_params:,} total "
        f"({trainable_params/total_params*100:.1f}%)"
    )

    # ===== 准备数据集 =====
    full_dataset = MusicSliceDataset(SLICES_DIR, MAX_DURATION, SAMPLE_RATE)

    # 划分训练集和验证集 (90/10)
    total = len(full_dataset)
    val_size = max(int(total * 0.1), 1)
    train_size = total - val_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    logger.info(f"Train: {train_size} samples, Eval: {val_size} samples")

    # ===== 数据整理函数 =====
    # MusicGen 需要 input_ids（文本 token）和 input_values（音频波形）
    # 使用占位文本 "music" 让 T5 编码器产生 conditioning embeddings
    DUMMY_PROMPT = "music"

    def data_collator(features):
        """将样本列表整理为 MusicGen 训练所需的批次格式。"""
        # Tokenize 占位文本 prompt
        text_inputs = processor.tokenizer(
            [DUMMY_PROMPT] * len(features),
            padding=True,
            return_tensors="pt",
        )

        # 堆叠音频波形，添加通道维度 [B, T] -> [B, 1, T]（单声道）
        input_values = torch.stack([f["input_values"] for f in features]).unsqueeze(1)

        # 构建正确的 attention_mask：text 部分
        text_attention_mask = text_inputs.attention_mask

        # audio padding mask: 全 1（所有音频都是完整 30s，无 padding）
        # 形状 [B, audio_length] 但太大，传 None 让模型内部处理
        return {
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_attention_mask,
            "input_values": input_values,
        }

    # ===== 训练配置 =====
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    total_steps = (train_size * NUM_EPOCHS) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    warmup_steps = int(total_steps * WARMUP_RATIO)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        fp16=(device.type == "cuda"),
        max_grad_norm=MAX_GRAD_NORM,
        logging_dir=str(LOG_DIR / "tensorboard"),
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_total_limit=3,
        dataloader_num_workers=0,
        seed=SEED,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    logger.info(f"Training config:")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} (accumulation)")
    logger.info(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"  Total steps: ~{total_steps}")
    logger.info(f"  Warmup steps: {warmup_steps}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")

    # ===== 创建 Trainer =====
    trainer = MusicGenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # ===== 开始训练 =====
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    start_time = time.time()

    train_result = trainer.train()

    elapsed = time.time() - start_time

    # ===== 保存最终模型 =====
    final_model_dir = OUTPUT_DIR / "final"
    trainer.save_model(str(final_model_dir))
    processor.save_pretrained(str(final_model_dir))

    # 保存训练元数据
    metadata = {
        "model": "facebook/musicgen-small",
        "epochs": NUM_EPOCHS,
        "train_samples": train_size,
        "eval_samples": val_size,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "total_steps": train_result.metrics.get("train_steps_per_second", 0),
        "train_loss": train_result.metrics.get("train_loss", 0),
        "elapsed_seconds": elapsed,
        "elapsed_formatted": f"{elapsed/60:.1f} minutes",
    }

    with open(OUTPUT_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # ===== 输出训练结果 =====
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)

    metrics = train_result.metrics
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

    logger.info(f"  Elapsed: {elapsed/60:.1f} minutes")
    logger.info(f"  Final model saved to: {final_model_dir.resolve()}")
    logger.info(f"  Training metadata: {OUTPUT_DIR / 'training_metadata.json'}")

    return metrics


if __name__ == "__main__":
    main()
