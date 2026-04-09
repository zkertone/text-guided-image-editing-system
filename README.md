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

如果你在 Colab 中运行，可以将 `launch()` 参数改成：

```python
demo.launch(share=True)
```

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

`.gitignore` 已忽略以下常见内容：

- 虚拟环境目录，如 `.venv/`
- Python 缓存，如 `__pycache__/`
- IDE 配置，如 `.idea/`
- macOS 系统文件，如 `.DS_Store`
- 本地输入输出图片，如 `data/input/`、`data/output/`
- 本地缓存目录

说明：
`data/input/` 和 `data/output/` 中的实际图片结果默认不提交到 Git，避免仓库越来越大。

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

如果你希望打开 Gradio 的公开分享链接，也可以使用：

```bash
GRADIO_SHARE=true python -m app.main
```

常用环境变量说明：

- `GRADIO_SERVER_NAME`：监听地址，服务器上通常设置为 `0.0.0.0`
- `GRADIO_SERVER_PORT`：服务端口，例如 `7860`
- `GRADIO_SHARE`：是否开启 Gradio 分享链接，`true` 或 `false`

访问方式说明：

- 本地运行时，通常直接访问终端输出的本地地址
- 服务器运行时，若安全组和防火墙已放行对应端口，可通过 `http://服务器IP:7860` 访问

## 7. Linux GPU 云服务器最小部署说明

以下流程适合 Ubuntu 等常见 Linux GPU 服务器。

### 7.1 克隆项目

```bash
git clone <your_repo_url>
cd <your_project_dir>
```

### 7.2 创建虚拟环境

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 7.3 安装依赖

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 7.4 启动项目

```bash
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 python -m app.main
```

### 7.5 访问 Gradio

如果服务器公网 IP 为 `1.2.3.4`，并且 `7860` 端口已放行，则可在浏览器中访问：

```text
http://1.2.3.4:7860
```

说明：

- 首次运行会下载模型，时间可能较长
- 若服务器未放行端口，则外部浏览器无法访问
- 若仅做临时演示，也可以尝试 `GRADIO_SHARE=true`

## 8. 当前 V1 功能

- 上传输入图像
- 输入英文编辑指令
- 调整推理步数
- 调整图像引导强度
- 调整文本引导强度
- 输出编辑后的图像
- 自动保存预处理后的输入图像到 `data/input/`
- 自动保存编辑结果到 `data/output/`

## 9. 模块说明

### `app/pipeline_loader.py`

负责加载预训练 InstructPix2Pix 模型和调度器，并自动选择运行设备。

### `app/editor.py`

负责封装图像预处理与图像编辑推理逻辑。

### `app/ui.py`

负责构建 Gradio 交互界面。

### `app/main.py`

项目主入口，负责初始化模型、编辑器和界面。

## 10. 注意事项

- 当前模型默认使用：`timbrooks/instruct-pix2pix`
- 输入图片会统一缩放到 `512 x 512`
- 文本指令建议优先使用英文，以获得更稳定效果
- 首次运行时会下载模型文件，耗时取决于网络环境
- 输入图和输出图会自动按时间戳命名保存，避免文件覆盖
- 服务器部署阶段建议优先使用 GPU 环境，CPU 仅适合基础验证

## 11. 后续迭代建议

当前建议按以下顺序逐步扩展：

1. 完善实验记录与截图整理
2. 补充 Git 提交历史和阶段性版本说明
3. 在服务器上完成一次完整演示验证
4. 再考虑增加局部编辑等后续功能
