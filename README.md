# Text-Guided Image Editing System

本科毕业设计题目：

《基于大模型的文字驱动图像编辑方法与系统实现》

本项目是一个面向演示与实验整理的 V1 工程版本，基于 Hugging Face Diffusers 和预训练 InstructPix2Pix 管线，实现最小可运行的文字驱动图像编辑系统。

## 1. 项目目标

当前版本聚焦以下内容：

- 使用预训练 `InstructPix2Pix` 完成文字驱动图像编辑
- 支持用户上传图片并输入编辑指令
- 使用 Gradio 提供可视化交互界面
- 将实验 notebook 中已验证的代码整理为清晰的工程结构

说明：
当前版本仅保留 V1 Demo 所需功能，不包含 Inpainting、ControlNet、数据库或用户系统等扩展模块。

## 2. 项目结构

```text
text-guided-image-editing-system/
├─ README.md
├─ requirements.txt
├─ notebooks/
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ pipeline_loader.py
│  ├─ editor.py
│  └─ ui.py
├─ data/
│  ├─ input/
│  └─ output/
└─ docs/
   └─ experiment_log.md
```

## 3. 环境准备

建议优先在 Colab 或云端 GPU 环境运行。

### 3.1 安装依赖

```bash
pip install -r requirements.txt
```

### 3.2 可选说明

- 若在 GPU 环境中运行，程序会优先使用 CUDA
- 若当前环境没有可用 CUDA，程序会自动回退到 CPU
- 由于扩散模型推理较慢，CPU 环境下仅适合做基础验证，不适合正式演示

## 4. 运行方式

在项目根目录下执行：

```bash
python -m app.main
```

运行后会启动 Gradio 本地界面。

如果你在 Colab 中运行，可以将 `launch()` 参数改成：

```python
demo.launch(share=True)
```

## 5. 当前 V1 功能

- 上传输入图像
- 输入英文编辑指令
- 调整推理步数
- 调整图像引导强度
- 调整文本引导强度
- 输出编辑后的图像
- 自动保存预处理后的输入图像到 `data/input/`
- 自动保存编辑结果到 `data/output/`

## 6. 模块说明

### `app/pipeline_loader.py`

负责加载预训练 InstructPix2Pix 模型和调度器，并自动选择运行设备。

### `app/editor.py`

负责封装图像预处理与图像编辑推理逻辑。

### `app/ui.py`

负责构建 Gradio 交互界面。

### `app/main.py`

项目主入口，负责初始化模型、编辑器和界面。

## 7. 后续迭代建议

当前建议按以下顺序逐步扩展：

1. 完成 V1 工程版的稳定运行与截图留档
2. 增加输入输出图片保存功能
3. 增加实验参数记录
4. 再考虑加入局部编辑或 Inpainting 功能

## 8. 注意事项

- 当前模型默认使用：`timbrooks/instruct-pix2pix`
- 输入图片会统一缩放到 `512 x 512`
- 文本指令建议优先使用英文，以获得更稳定效果
- 首次运行时会下载模型文件，耗时取决于网络环境
- 输入图和输出图会自动按时间戳命名保存，避免文件覆盖
