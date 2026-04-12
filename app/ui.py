import gradio as gr


def create_ui(image_editor):
    """Create the Gradio interface for the V1 demo."""
    global_example_prompts = [
        "make the hair blonde",
        "change the background to a beach",
        "make the sky sunset orange",
        "make the suit blue",
    ]
    local_example_prompts = [
        "change the hair color to natural blonde",
        "change the suit color to blue",
        "change only the selected area to red",
        "make the masked area darker",
    ]

    def update_ui_by_mode(mode, mask_source):
        is_local = mode == "local_inpaint"
        show_upload_mask = is_local and mask_source == "uploaded_mask"
        show_drawn_mask = is_local and mask_source == "drawn_mask"
        return (
            gr.update(visible=is_local),
            gr.update(visible=show_upload_mask),
            gr.update(visible=show_drawn_mask),
            gr.update(visible=not is_local),
            gr.update(visible=not is_local),
            gr.update(visible=is_local),
            gr.update(visible=is_local),
        )

    def sync_editor_image(input_image):
        if input_image is None:
            return None
        return input_image

    def run_edit(
        mode,
        input_image,
        mask_source,
        uploaded_mask_image,
        drawn_mask_data,
        prompt,
        steps,
        image_guidance,
        text_guidance,
    ):
        if input_image is None:
            return None, "请先上传一张输入图片。"

        if not prompt or not prompt.strip():
            return None, "请输入英文编辑指令。"

        if mode == "local_inpaint":
            if mask_source == "uploaded_mask" and uploaded_mask_image is None:
                return None, "局部编辑模式下，请上传黑白 Mask 图。"
            if mask_source == "drawn_mask" and drawn_mask_data is None:
                return None, "局部编辑模式下，请先在线绘制 Mask。"

        result = image_editor.edit_image(
            input_image=input_image,
            prompt=prompt,
            mode=mode,
            mask_source=mask_source,
            mask_image=uploaded_mask_image,
            drawn_mask_data=drawn_mask_data,
            num_inference_steps=steps,
            image_guidance_scale=image_guidance,
            guidance_scale=text_guidance,
        )

        return result["result_image"], result["summary_text"]

    with gr.Blocks() as demo:
        gr.Markdown("# 文字驱动图像编辑 Demo")
        gr.Markdown(
            "本系统支持整体编辑、上传 Mask 图的局部编辑，以及在线绘制 Mask 的局部编辑。"
        )

        with gr.Row():
            with gr.Column(scale=5):
                mode = gr.Radio(
                    choices=[
                        ("整体编辑", "global_edit"),
                        ("局部编辑", "local_inpaint"),
                    ],
                    value="global_edit",
                    label="编辑模式",
                )
                input_image = gr.Image(type="pil", label="输入图像")

                with gr.Group(visible=False) as local_group:
                    gr.Markdown("### 局部编辑区域")
                    mask_source = gr.Radio(
                        choices=[
                            ("上传 Mask 图", "uploaded_mask"),
                            ("在线绘制 Mask", "drawn_mask"),
                        ],
                        value="uploaded_mask",
                        label="Mask 来源",
                    )
                    uploaded_mask = gr.Image(
                        type="pil",
                        label="上传黑白 Mask 图",
                        visible=True,
                    )
                    drawn_mask = gr.ImageEditor(
                        type="pil",
                        label="在线绘制 Mask",
                        brush=gr.Brush(colors=["#FFFFFF"], color_mode="fixed"),
                        visible=False,
                    )

                prompt = gr.Textbox(
                    label="编辑指令（英文）",
                    placeholder="例如：make the sky sunset orange",
                )
                steps = gr.Slider(10, 40, value=20, step=1, label="推理步数")
                image_guidance_markdown = gr.Markdown("### 整体编辑参数")
                image_guidance = gr.Slider(
                    1.0,
                    2.5,
                    value=1.5,
                    step=0.1,
                    label="图像引导强度",
                    visible=True,
                )
                text_guidance = gr.Slider(
                    5.0, 10.0, value=7.5, step=0.5, label="文本引导强度"
                )
                run_button = gr.Button("开始编辑", variant="primary")

            with gr.Column(scale=5):
                overall_help = gr.Markdown(
                    "### 整体编辑如何使用\n"
                    "1. 选择“整体编辑”。\n"
                    "2. 上传输入图像。\n"
                    "3. 输入英文编辑指令。\n"
                    "4. 调整推理步数、图像引导强度和文本引导强度。\n"
                    "5. 点击“开始编辑”查看结果。\n\n"
                    "整体编辑示例 Prompt：\n"
                    f"1. {global_example_prompts[0]}\n"
                    f"2. {global_example_prompts[1]}\n"
                    f"3. {global_example_prompts[2]}\n"
                    f"4. {global_example_prompts[3]}",
                    visible=True,
                )
                local_help = gr.Markdown(
                    "### 局部编辑如何使用\n"
                    "1. 选择“局部编辑”。\n"
                    "2. 上传输入图像。\n"
                    "3. 选择 Mask 来源：上传 Mask 图或在线绘制 Mask。\n"
                    "4. 输入英文编辑指令。\n"
                    "5. 点击“开始编辑”查看结果。\n\n"
                    "Mask 使用说明：\n"
                    "- 白色区域表示需要编辑。\n"
                    "- 黑色区域表示保持不变。\n"
                    "- 建议先使用较小 Mask 区域做测试。\n"
                    "- 建议使用更具体、更自然的英文 Prompt。\n\n"
                    "局部编辑推荐 Prompt：\n"
                    f"1. {local_example_prompts[0]}\n"
                    f"2. {local_example_prompts[1]}\n"
                    f"3. {local_example_prompts[2]}\n"
                    f"4. {local_example_prompts[3]}",
                    visible=False,
                )
                mask_area_help = gr.Markdown(
                    "### Mask 区域说明\n"
                    "当前处于局部编辑模式。\n"
                    "- 若选择“上传 Mask 图”，请上传黑白 Mask 图。\n"
                    "- 若选择“在线绘制 Mask”，请在原图上直接涂抹待编辑区域。\n"
                    "- 上传 Mask 图与在线绘制 Mask 都会统一转换为标准黑白 Mask 后进入 Inpainting 流程。",
                    visible=False,
                )
                output_image = gr.Image(type="pil", label="输出结果")
                info_text = gr.Textbox(label="实验信息", lines=13)

        mode.change(
            fn=update_ui_by_mode,
            inputs=[mode, mask_source],
            outputs=[
                local_group,
                uploaded_mask,
                drawn_mask,
                image_guidance_markdown,
                image_guidance,
                overall_help,
                local_help,
            ],
        ).then(
            fn=lambda selected_mode: gr.update(visible=selected_mode == "local_inpaint"),
            inputs=mode,
            outputs=mask_area_help,
        )

        mask_source.change(
            fn=update_ui_by_mode,
            inputs=[mode, mask_source],
            outputs=[
                local_group,
                uploaded_mask,
                drawn_mask,
                image_guidance_markdown,
                image_guidance,
                overall_help,
                local_help,
            ],
        ).then(
            fn=lambda selected_mode: gr.update(visible=selected_mode == "local_inpaint"),
            inputs=mode,
            outputs=mask_area_help,
        )

        input_image.change(
            fn=sync_editor_image,
            inputs=input_image,
            outputs=drawn_mask,
        )

        run_button.click(
            fn=run_edit,
            inputs=[
                mode,
                input_image,
                mask_source,
                uploaded_mask,
                drawn_mask,
                prompt,
                steps,
                image_guidance,
                text_guidance,
            ],
            outputs=[output_image, info_text],
        )

    return demo
