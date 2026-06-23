"""示例：下载模型并生成音乐

该脚本演示了完整的工作流：
1. 从 Hugging Face Hub 下载 MusicGen 模型
2. 加载模型
3. 使用不同参数生成音乐
4. 保存生成的音频文件
"""

import sys
from pathlib import Path

# 确保可以导入项目模块
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_manager import ModelManager
from generation.generator import MusicGenerator
from utils.logger import setup_logger


def main():
    setup_logger(log_level="INFO")

    # ===== 步骤 1: 下载模型 =====
    print("=" * 60)
    print("步骤 1: 检查并下载模型")
    print("=" * 60)

    manager = ModelManager()

    # 列出可用模型
    models = manager.list_available_models()
    print(f"\n支持 {len(models)} 个模型:")
    for m in models:
        status = "[已缓存]" if m["cached"] else "[未下载]"
        print(f"  {status} {m['name']} - {m['description']}")

    # 下载 MusicGen Small
    model_name = "musicgen-small"
    model_path = manager.get_model_path(model_name)

    if model_path is None:
        print(f"\n正在下载 {model_name}...")
        model_path = manager.download_model(model_name)

    print(f"模型路径: {model_path}")

    # ===== 步骤 2: 创建生成器 =====
    print("\n" + "=" * 60)
    print("步骤 2: 初始化音乐生成器")
    print("=" * 60)

    generator = MusicGenerator(
        architecture="musicgen",
        model_name=model_name,
    )
    generator.load_model(model_path)

    info = generator.get_model_info()
    print(f"模型信息: {info}")

    # ===== 步骤 3: 生成音乐 =====
    print("\n" + "=" * 60)
    print("步骤 3: 生成音乐样本")
    print("=" * 60)

    output_dir = Path("./outputs/examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 示例 1: 基本文本生成
    prompts = [
        ("一段欢快的钢琴曲", 15),
        ("舒缓的大提琴独奏", 20),
        ("电子舞曲，节奏明快", 15),
    ]

    for i, (prompt, duration) in enumerate(prompts):
        print(f"\n生成样本 {i+1}: '{prompt}' ({duration}秒)")

        audio = generator.generate(
            prompt=prompt,
            duration=duration,
            temperature=1.0,
            top_k=250,
            guidance_scale=3.0,
        )

        output_path = output_dir / f"example_{i+1}.wav"
        generator.save_audio(audio, output_path)
        print(f"  已保存: {output_path}")

    # 示例 2: 不同温度参数对比
    print("\n" + "=" * 60)
    print("步骤 4: 温度参数对比")
    print("=" * 60)

    test_prompt = "古典风格的弦乐四重奏"
    temperatures = [0.5, 1.0, 1.5]

    for temp in temperatures:
        print(f"\n温度={temp}: 生成 '{test_prompt}'")

        audio = generator.generate(
            prompt=test_prompt,
            duration=15,
            temperature=temp,
        )

        output_path = output_dir / f"temp_{temp}.wav"
        generator.save_audio(audio, output_path)
        print(f"  已保存: {output_path}")

    print("\n" + "=" * 60)
    print("所有样本生成完成!")
    print(f"输出目录: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
