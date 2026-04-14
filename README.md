# Text-Guided Image Editing System

本科毕业设计题目：

《基于大模型的文字驱动图像编辑方法与系统实现》

本项目是一个面向演示与实验整理的 V1 工程版本，基于 Hugging Face Diffusers 和预训练 InstructPix2Pix 管线，实现最小可运行的文字驱动图像编辑系统。

## 1. 项目目标

当前版本聚焦以下内容：

- 使用预训练 `InstructPix2Pix` 完成文字驱动图像编辑
- 使用预训练 `Stable Diffusion Inpainting` 实现基于 Mask 的局部编辑
- 使用 `Canny ControlNet` 实现结构保持编辑
- 支持用户上传图片并输入编辑指令
- 使用 Gradio 提供可视化交互界面
- 支持上传 Mask 图与在线绘制 Mask 两种局部编辑方式
- 将实验 notebook 中已验证的代码整理为清晰的工程结构

说明：
当前版本聚焦课程题目相关功能，不包含 ControlNet、数据库或用户系统等扩展模块。

## 2. 项目结构

```text
text-guided-image-editing-system/
├─ .gitignore
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

## 4. 本地运行方式

在项目根目录下执行：

```bash
python -m app.main
```

运行后会启动 Gradio 本地界面。

如果你希望通过 Google Colab 启动整个项目，可以使用：

- [notebooks/colab_run_project.ipynb](/Users/zjx/Documents/New project/notebooks/colab_run_project.ipynb)

## 5. Git 管理建议

建议先初始化仓库并提交当前版本：

```bash
git init
git add .
git commit -m "init v1 project"
```

日常开发可以使用下面的最小工作流：

```bash
git status
git add .
git commit -m "update demo"
```

## 6. 远端 GPU 服务器运行方式

为了尽量少改代码，当前项目支持通过环境变量控制 Gradio 启动参数。

默认本地运行：

```bash
python -m app.main
```

如果在 Linux GPU 服务器上运行，可使用：

```bash
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 python -m app.main
```

## 7. 当前 V1 功能

- 上传输入图像
- 上传黑白 Mask 图进行局部编辑
- 支持在线绘制 Mask 进行局部编辑
- 输入英文编辑指令
- 支持“整体编辑”“局部编辑”“结构保持编辑”三种模式
- 局部编辑支持“上传Mask图”与“在线绘制Mask”两种来源
- 结构保持编辑自动生成并展示 Canny 控制图
- 调整推理步数
- 调整图像引导强度
- 调整文本引导强度
- 输出编辑后的图像
- 自动保存预处理后的输入图像到 `data/input/`
- 自动保存编辑结果到 `data/output/`
- 自动追加实验记录到 `docs/experiment_log.csv`

## 8. 模块说明

### `app/pipeline_loader.py`

负责加载整体编辑、局部编辑与结构保持编辑所需的预训练管线，并自动选择运行设备。

### `app/editor.py`

负责封装图像预处理、局部 Mask 预处理、Canny 图生成、编辑推理逻辑与实验日志记录逻辑。

### `app/ui.py`

负责构建 Gradio 交互界面。

### `app/main.py`

项目主入口，负责初始化模型、编辑器和界面。

## 9. 注意事项

- 整体编辑模型默认使用：`timbrooks/instruct-pix2pix`
- 局部编辑模型默认使用：`runwayml/stable-diffusion-inpainting`
- 结构保持编辑基础模型默认使用：`runwayml/stable-diffusion-v1-5`
- Canny ControlNet 模型默认使用：`lllyasviel/sd-controlnet-canny`
- 输入图片会统一缩放到 `512 x 512`
- 局部编辑时，mask 图会统一缩放到 `512 x 512`
- 局部编辑时，白色区域表示需要编辑，黑色区域表示保持不变
- 在线绘制 Mask 时，系统会自动将绘制区域转换为标准黑白 mask
- 结构保持编辑会自动根据输入图像生成 Canny 边缘图作为控制条件
- 文本指令建议优先使用英文，以获得更稳定效果
- 首次运行时会下载模型文件，耗时取决于网络环境
- 输入图和输出图会自动按时间戳命名保存，避免文件覆盖
- 服务器部署阶段建议优先使用 GPU 环境，CPU 仅适合基础验证
