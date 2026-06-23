"""Gradio Web UI：用户友好的音乐生成界面。

提供基于 Gradio 的交互式 Web 界面，支持：
- 文本提示生成音乐
- 风格参数调节
- 音频播放和下载
- 训练数据管理
"""

import sys
from pathlib import Path

# 将项目根目录添加到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
from loguru import logger


def create_app():
    """创建 Gradio Web UI 应用。

    Returns:
        gr.Blocks: Gradio 应用实例。
    """

    # 全局生成器实例（延迟初始化）
    generator_state = {"generator": None}

    def init_generator(architecture, model_name):
        """初始化或切换生成器。"""
        try:
            from generation.generator import MusicGenerator
            from models.model_manager import ModelManager

            generator = MusicGenerator(
                architecture=architecture,
                model_name=model_name,
            )

            manager = ModelManager()
            model_path = manager.get_model_path(model_name)

            if model_path is None:
                yield "模型未缓存，正在下载...", None
                model_path = manager.download_model(model_name)

            generator.load_model(model_path)
            generator_state["generator"] = generator

            info = generator.get_model_info()
            info_text = (
                f"模型: {info.get('name', 'N/A')}\n"
                f"架构: {info.get('architecture', 'N/A')}\n"
                f"参数: {info.get('total', 0):,}\n"
                f"设备: {info.get('device', 'N/A')}"
            )
            yield f"模型加载完成\n\n{info_text}", None

        except Exception as e:
            yield f"模型加载失败: {e}", None

    def generate_music(
        prompt, duration, temperature, top_k, guidance_scale, style_desc
    ):
        """生成音乐。"""
        if generator_state["generator"] is None:
            return None, "请先加载模型"

        try:
            audio = generator_state["generator"].generate(
                prompt=prompt if prompt else None,
                duration=duration,
                temperature=temperature,
                top_k=int(top_k),
                guidance_scale=guidance_scale,
                style_description=style_desc if style_desc else None,
            )

            sample_rate = 32000
            status = f"生成完成: {len(audio)/sample_rate:.1f} 秒"
            return (sample_rate, audio), status

        except Exception as e:
            return None, f"生成失败: {e}"

    def analyze_style(audio_file):
        """分析上传音频的风格。"""
        if audio_file is None:
            return "请上传音频文件"

        try:
            from dataset.feature_extractor import StyleFeatureExtractor

            extractor = StyleFeatureExtractor()
            features = extractor.extract_from_file(audio_file)
            summary = features.get("summary", {})

            result_lines = ["=== 风格分析结果 ===\n"]
            for key, value in summary.items():
                if isinstance(value, float):
                    result_lines.append(f"{key}: {value:.4f}")
                else:
                    result_lines.append(f"{key}: {value}")

            return "\n".join(result_lines)
        except Exception as e:
            return f"分析失败: {e}"

    def import_audio(audio_file):
        """导入音频到数据集。"""
        if audio_file is None:
            return "请上传音频文件"

        try:
            from dataset.dataset_manager import MusicDatasetManager

            manager = MusicDatasetManager()
            import shutil

            raw_dir = Path(manager.dataset_dir) / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            filename = Path(audio_file).name
            target = raw_dir / filename
            shutil.copy2(audio_file, target)

            return f"已导入: {filename}"
        except Exception as e:
            return f"导入失败: {e}"

    # ===== 构建 UI 界面 =====

    with gr.Blocks(
        title="QS Music - 音乐风格迁移与生成",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown(
            """
            # QS Music - 音乐风格迁移与生成系统
            基于深度学习的音乐风格学习与生成平台
            """
        )

        with gr.Tabs():
            # ===== Tab 1: 音乐生成 =====
            with gr.TabItem("音乐生成"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 模型设置")
                        architecture = gr.Dropdown(
                            choices=["musicgen", "musecoco"],
                            value="musicgen",
                            label="模型架构",
                        )
                        model_name = gr.Dropdown(
                            choices=[
                                "musicgen-small",
                                "musicgen-medium",
                                "musicgen-large",
                                "musicgen-melody",
                                "musecoco",
                            ],
                            value="musicgen-small",
                            label="模型选择",
                        )
                        load_btn = gr.Button("加载模型", variant="primary")
                        model_status = gr.Textbox(
                            label="模型状态", interactive=False
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### 生成设置")
                        prompt = gr.Textbox(
                            label="文本提示",
                            placeholder="描述你想生成的音乐，如：一段欢快的钢琴曲，带有爵士风格",
                            lines=2,
                        )
                        style_desc = gr.Textbox(
                            label="风格描述（可选）",
                            placeholder="如：古典、爵士、电子、民谣...",
                        )

                        with gr.Row():
                            duration = gr.Slider(
                                minimum=5,
                                maximum=120,
                                value=30,
                                step=5,
                                label="时长（秒）",
                            )
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="温度",
                            )
                        with gr.Row():
                            top_k = gr.Slider(
                                minimum=10,
                                maximum=500,
                                value=250,
                                step=10,
                                label="Top-k",
                            )
                            guidance_scale = gr.Slider(
                                minimum=1.0,
                                maximum=10.0,
                                value=3.0,
                                step=0.5,
                                label="Guidance Scale",
                            )

                        generate_btn = gr.Button(
                            "生成音乐", variant="primary", size="lg"
                        )
                        gen_status = gr.Textbox(
                            label="生成状态", interactive=False
                        )

                audio_output = gr.Audio(
                    label="生成的音乐", type="numpy"
                )

                # 绑定事件
                load_btn.click(
                    init_generator,
                    inputs=[architecture, model_name],
                    outputs=[model_status, audio_output],
                )
                generate_btn.click(
                    generate_music,
                    inputs=[
                        prompt, duration, temperature,
                        top_k, guidance_scale, style_desc,
                    ],
                    outputs=[audio_output, gen_status],
                )

            # ===== Tab 2: 风格分析 =====
            with gr.TabItem("风格分析"):
                gr.Markdown("### 上传音频文件进行风格分析")
                audio_input = gr.Audio(label="上传音频", type="filepath")
                analyze_btn = gr.Button("分析风格")
                analysis_result = gr.Textbox(
                    label="分析结果", lines=15, interactive=False
                )
                analyze_btn.click(
                    analyze_style,
                    inputs=[audio_input],
                    outputs=[analysis_result],
                )

            # ===== Tab 3: 数据管理 =====
            with gr.TabItem("数据管理"):
                gr.Markdown("### 导入音乐到训练数据集")
                data_input = gr.Audio(label="上传音频文件", type="filepath")
                import_btn = gr.Button("导入到数据集")
                import_status = gr.Textbox(
                    label="导入状态", interactive=False
                )
                import_btn.click(
                    import_audio,
                    inputs=[data_input],
                    outputs=[import_status],
                )

            # ===== Tab 4: 系统信息 =====
            with gr.TabItem("系统信息"):
                gr.Markdown("### 系统环境信息")
                info_btn = gr.Button("获取系统信息")
                info_output = gr.Textbox(
                    label="系统信息", lines=20, interactive=False
                )

                def get_sys_info():
                    try:
                        from utils.device import get_device_info
                        info = get_device_info()
                        return "\n".join(f"{k}: {v}" for k, v in info.items())
                    except Exception as e:
                        return f"获取信息失败: {e}"

                info_btn.click(get_sys_info, outputs=[info_output])

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_port=7860, share=False)
