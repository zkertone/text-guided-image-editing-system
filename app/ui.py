import gradio as gr


def create_ui(image_editor):
    """Create the Gradio interface for the V1 demo."""

    def run_edit(
        input_image,
        prompt,
        steps,
        image_guidance,
        text_guidance,
    ):
        if input_image is None:
            return None, ""

        result = image_editor.edit_image(
            input_image=input_image,
            prompt=prompt,
            num_inference_steps=steps,
            image_guidance_scale=image_guidance,
            guidance_scale=text_guidance,
        )

        info_text = (
            f"输入图保存路径: {result['input_save_path']}\n"
            f"输出图保存路径: {result['output_save_path']}\n\n"
            f"{result['summary_text']}"
        )

        return result["result_image"], info_text

    demo = gr.Interface(
        fn=run_edit,
        inputs=[
            gr.Image(type="pil", label="上传图片"),
            gr.Textbox(
                label="编辑指令（英文）",
                placeholder="例如：make the sky sunset orange",
            ),
            gr.Slider(10, 40, value=20, step=1, label="推理步数"),
            gr.Slider(1.0, 2.5, value=1.5, step=0.1, label="图像引导强度"),
            gr.Slider(5.0, 10.0, value=7.5, step=0.5, label="文本引导强度"),
        ],
        outputs=[
            gr.Image(type="pil", label="编辑结果"),
            gr.Textbox(label="实验信息", lines=10),
        ],
        title="文字驱动图像编辑 Demo",
        description="上传图片并输入英文编辑指令，系统返回编辑后的图像，并展示保存路径与参数摘要。",
    )

    return demo
