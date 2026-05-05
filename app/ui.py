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
    control_example_prompts = [
        "make the building more futuristic",
        "turn the road into a snowy path",
        "make the room look modern",
        "change the car to a red sports car",
    ]

    def update_ui_by_mode(mode, mask_source):
        is_local = mode == "local_inpaint"
        is_control = mode == "controlnet_canny"
        show_upload_mask = is_local and mask_source == "uploaded_mask"
        show_drawn_mask = is_local and mask_source == "drawn_mask"
        show_image_guidance = mode == "global_edit"
        return (
            gr.update(visible=is_local),
            gr.update(visible=show_upload_mask),
            gr.update(visible=show_drawn_mask),
            gr.update(visible=show_image_guidance),
            gr.update(visible=show_image_guidance),
            gr.update(visible=mode == "global_edit"),
            gr.update(visible=is_local),
            gr.update(visible=is_control),
            gr.update(visible=is_control),
            gr.update(visible=is_control),
        )

    def sync_editor_image(input_image):
        if input_image is None:
            return None
        return input_image

    def get_record_id(value):
        if value is None:
            raise ValueError("请输入记录 ID。")
        return int(value)

    def get_image_id(value):
        if value is None:
            raise ValueError("请输入图片 ID。")
        return int(value)

    def format_record_detail(record):
        if record is None:
            return "未找到该记录，或该记录已被删除。"

        fields = [
            ("id", record.get("id")),
            ("created_at", record.get("created_at")),
            ("mode", record.get("mode")),
            ("prompt", record.get("prompt")),
            ("mask_source", record.get("mask_source")),
            ("control_type", record.get("control_type")),
            ("num_inference_steps", record.get("num_inference_steps")),
            ("image_guidance_scale", record.get("image_guidance_scale")),
            ("guidance_scale", record.get("guidance_scale")),
            ("input_image_id", record.get("input_image_id")),
            ("mask_image_id", record.get("mask_image_id")),
            ("control_image_id", record.get("control_image_id")),
            ("output_image_id", record.get("output_image_id")),
            ("status", record.get("status")),
            ("error_message", record.get("error_message") or ""),
        ]
        return "\n".join(f"{key}: {value}" for key, value in fields)

    def view_history_record(record_id):
        try:
            record_id = get_record_id(record_id)
            record = image_editor.history_manager.get_record_detail(record_id)
            images = image_editor.history_manager.get_record_images(record_id)
            return (
                images["input"],
                images["mask"],
                images["control"],
                images["output"],
                format_record_detail(record),
            )
        except Exception as error:
            return None, None, None, None, f"查看记录失败: {error}"

    def load_history_input_image(image_id):
        try:
            image_id = get_image_id(image_id)
            image = image_editor.history_manager.load_image(image_id)
            return (
                image,
                image,
                f"已加载历史输入图，image_id={image_id}",
                image_editor.history_manager.get_recent_input_images(),
            )
        except Exception as error:
            return (
                gr.update(),
                gr.update(),
                f"加载历史输入图失败: {error}",
                image_editor.history_manager.get_recent_input_images(),
            )

    def delete_history_record(record_id):
        try:
            record_id = get_record_id(record_id)
            deleted = image_editor.history_manager.soft_delete_record(record_id)
            if deleted:
                message = f"已逻辑删除编辑记录，record_id={record_id}"
            else:
                message = "未找到该记录，或该记录已被删除。"
            return message, image_editor.get_recent_records()
        except Exception as error:
            return f"删除记录失败: {error}", image_editor.get_recent_records()

    def delete_history_image(image_id):
        try:
            image_id = get_image_id(image_id)
            deleted = image_editor.history_manager.soft_delete_image(image_id)
            if deleted:
                message = f"已逻辑删除图片，image_id={image_id}"
            else:
                message = "未找到该图片，或该图片已被删除。"
            return (
                message,
                image_editor.get_recent_records(),
                image_editor.history_manager.get_recent_input_images(),
            )
        except Exception as error:
            return (
                f"删除图片失败: {error}",
                image_editor.get_recent_records(),
                image_editor.history_manager.get_recent_input_images(),
            )

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
        recent_records = image_editor.get_recent_records()
        recent_input_images = image_editor.history_manager.get_recent_input_images()

        if input_image is None:
            return None, None, "请先上传一张输入图片。", recent_records, recent_input_images

        if not prompt or not prompt.strip():
            return None, None, "请输入英文编辑指令。", recent_records, recent_input_images

        if mode == "local_inpaint":
            if mask_source == "uploaded_mask" and uploaded_mask_image is None:
                return (
                    None,
                    None,
                    "局部编辑模式下，请上传黑白 Mask 图。",
                    recent_records,
                    recent_input_images,
                )
            if mask_source == "drawn_mask" and drawn_mask_data is None:
                return (
                    None,
                    None,
                    "局部编辑模式下，请先在线绘制 Mask。",
                    recent_records,
                    recent_input_images,
                )

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

        return (
            result["result_image"],
            result["control_image"],
            result["summary_text"],
            image_editor.get_recent_records(),
            image_editor.history_manager.get_recent_input_images(),
        )

    with gr.Blocks() as demo:
        gr.Markdown("# 文字驱动图像编辑 Demo")
        gr.Markdown(
            "本系统支持整体编辑、局部编辑，以及基于 Canny ControlNet 的结构保持编辑。"
        )

        with gr.Row():
            with gr.Column(scale=5):
                mode = gr.Radio(
                    choices=[
                        ("整体编辑", "global_edit"),
                        ("局部编辑", "local_inpaint"),
                        ("结构保持编辑", "controlnet_canny"),
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
                control_help = gr.Markdown(
                    "### 结构保持编辑如何使用\n"
                    "1. 选择“结构保持编辑”。\n"
                    "2. 上传输入图像。\n"
                    "3. 输入英文编辑指令。\n"
                    "4. 系统会自动从原图生成 Canny 边缘图。\n"
                    "5. 使用该边缘图作为 ControlNet 条件输入生成结果。\n\n"
                    "与整体编辑相比，结构保持编辑更强调保留原图结构轮廓。\n"
                    "与局部编辑相比，结构保持编辑不需要上传或绘制 Mask。\n\n"
                    "结构保持编辑示例 Prompt：\n"
                    f"1. {control_example_prompts[0]}\n"
                    f"2. {control_example_prompts[1]}\n"
                    f"3. {control_example_prompts[2]}\n"
                    f"4. {control_example_prompts[3]}",
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
                canny_help = gr.Markdown(
                    "### Canny 控制图说明\n"
                    "当前处于结构保持编辑模式。\n"
                    "- 系统会根据输入图像自动提取边缘结构。\n"
                    "- Canny 控制图用于增强生成结果与原始结构的一致性。\n"
                    "- 该模式更适合在保留场景结构的前提下做内容变化。",
                    visible=False,
                )
                canny_preview = gr.Image(
                    type="pil",
                    label="Canny 控制图",
                    visible=False,
                )
                output_image = gr.Image(type="pil", label="输出结果")
                info_text = gr.Textbox(label="实验信息", lines=15)
                recent_records = gr.Dataframe(
                    headers=["id", "created_at", "mode", "prompt", "status"],
                    value=image_editor.get_recent_records(),
                    datatype=["number", "str", "str", "str", "str"],
                    row_count=(10, "fixed"),
                    col_count=(5, "fixed"),
                    label="最近编辑记录",
                    interactive=False,
                )

        with gr.Accordion("历史记录管理", open=False):
            gr.Markdown("### 查看历史记录")
            with gr.Row():
                view_record_id = gr.Number(label="编辑记录 ID", precision=0)
                view_record_button = gr.Button("查看记录")

            with gr.Row():
                history_input_image = gr.Image(type="pil", label="历史输入图")
                history_mask_image = gr.Image(type="pil", label="历史 Mask 图")
                history_control_image = gr.Image(type="pil", label="历史 Canny 图")
                history_output_image = gr.Image(type="pil", label="历史输出图")

            history_detail_text = gr.Textbox(label="历史记录详情", lines=14)

            gr.Markdown("### 复用历史输入图")
            recent_input_images = gr.Dataframe(
                headers=["id", "file_name", "created_at", "width", "height"],
                value=image_editor.history_manager.get_recent_input_images(),
                datatype=["number", "str", "str", "number", "number"],
                row_count=(10, "fixed"),
                col_count=(5, "fixed"),
                label="最近输入图片",
                interactive=False,
            )
            with gr.Row():
                load_image_id = gr.Number(label="输入图片 ID", precision=0)
                load_image_button = gr.Button("加载为当前输入图")

            gr.Markdown("### 逻辑删除")
            with gr.Row():
                delete_record_id = gr.Number(label="删除记录 ID", precision=0)
                delete_record_button = gr.Button("删除记录")
                delete_image_id = gr.Number(label="删除图片 ID", precision=0)
                delete_image_button = gr.Button("删除图片")

            history_status = gr.Textbox(label="历史记录操作提示", lines=3)

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
                control_help,
                mask_area_help,
                canny_help,
            ],
        ).then(
            fn=lambda selected_mode: gr.update(visible=selected_mode == "controlnet_canny"),
            inputs=mode,
            outputs=canny_preview,
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
                control_help,
                mask_area_help,
                canny_help,
            ],
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
            outputs=[
                output_image,
                canny_preview,
                info_text,
                recent_records,
                recent_input_images,
            ],
        )

        view_record_button.click(
            fn=view_history_record,
            inputs=view_record_id,
            outputs=[
                history_input_image,
                history_mask_image,
                history_control_image,
                history_output_image,
                history_detail_text,
            ],
        )

        load_image_button.click(
            fn=load_history_input_image,
            inputs=load_image_id,
            outputs=[
                input_image,
                drawn_mask,
                history_status,
                recent_input_images,
            ],
        )

        delete_record_button.click(
            fn=delete_history_record,
            inputs=delete_record_id,
            outputs=[history_status, recent_records],
        )

        delete_image_button.click(
            fn=delete_history_image,
            inputs=delete_image_id,
            outputs=[history_status, recent_records, recent_input_images],
        )

    return demo
