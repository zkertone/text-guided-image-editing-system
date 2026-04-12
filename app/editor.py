import csv
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageChops


class ImageEditor:
    """A lightweight wrapper around the diffusion pipelines."""

    def __init__(self, pipeline, image_size=(512, 512)):
        if isinstance(pipeline, dict):
            self.pipelines = pipeline
        else:
            self.pipelines = {"global_edit": pipeline}

        self.image_size = image_size
        project_root = Path(__file__).resolve().parents[1]
        self.input_dir = project_root / "data" / "input"
        self.output_dir = project_root / "data" / "output"
        self.log_csv_path = project_root / "docs" / "experiment_log.csv"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)

    def preprocess_image(self, input_image: Image.Image) -> Image.Image:
        """Convert input image to RGB and resize it to the demo size."""
        return input_image.convert("RGB").resize(self.image_size)

    def preprocess_mask(self, mask_image: Image.Image) -> Image.Image:
        """Convert mask image to black-white format for inpainting."""
        mask = mask_image.convert("L").resize(self.image_size)
        return mask.point(lambda p: 255 if p >= 128 else 0)

    def extract_drawn_mask(self, editor_data) -> Image.Image:
        """Convert Gradio ImageEditor output into a standard black-white mask."""
        if editor_data is None:
            raise ValueError("Drawn mask data cannot be None.")

        if isinstance(editor_data, Image.Image):
            return self.preprocess_mask(editor_data)

        if not isinstance(editor_data, dict):
            raise ValueError("Unsupported drawn mask data format.")

        layers = editor_data.get("layers") or []
        if layers:
            merged = Image.new("RGBA", layers[0].size, (0, 0, 0, 0))
            for layer in layers:
                if layer is not None:
                    merged.alpha_composite(layer.convert("RGBA"))
            alpha_mask = merged.getchannel("A")
            return self.preprocess_mask(alpha_mask)

        background = editor_data.get("background")
        composite = editor_data.get("composite")
        if background is not None and composite is not None:
            background_rgba = background.convert("RGBA").resize(self.image_size)
            composite_rgba = composite.convert("RGBA").resize(self.image_size)
            diff_mask = ImageChops.difference(background_rgba, composite_rgba).convert("L")
            return self.preprocess_mask(diff_mask)

        if composite is not None:
            return self.preprocess_mask(composite)

        raise ValueError("Unable to extract a valid mask from drawn mask data.")

    def get_pipeline(self, mode: str):
        """Get the pipeline by current editing mode."""
        if mode not in self.pipelines:
            raise ValueError(f"Unsupported mode: {mode}")
        return self.pipelines[mode]

    def get_mode_label(self, mode: str) -> str:
        """Get a readable label for the current editing mode."""
        if mode == "local_inpaint":
            return "局部编辑"
        return "整体编辑"

    def get_mask_source_label(self, mask_source: str) -> str:
        """Get a readable label for the current mask source."""
        if mask_source == "uploaded_mask":
            return "上传Mask图"
        if mask_source == "drawn_mask":
            return "在线绘制"
        return "不适用"

    def get_model_name(self, mode: str) -> str:
        """Get the current pipeline model name for display."""
        pipeline = self.get_pipeline(mode)
        if hasattr(pipeline, "config") and hasattr(pipeline.config, "_name_or_path"):
            return str(pipeline.config._name_or_path)
        return "unknown"

    def get_device_name(self, mode: str) -> str:
        """Get the current execution device for display."""
        pipeline = self.get_pipeline(mode)
        if hasattr(pipeline, "_execution_device"):
            return str(pipeline._execution_device)
        if hasattr(pipeline, "device"):
            return str(pipeline.device)
        return "unknown"

    def _generate_timestamp(self) -> str:
        """Generate a timestamp-based filename suffix to avoid overwriting."""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _save_image(self, image: Image.Image, save_dir: Path, prefix: str) -> Path:
        """Save an image to the target directory with a timestamp filename."""
        filename = f"{prefix}_{self._generate_timestamp()}.png"
        save_path = save_dir / filename
        image.save(save_path)
        return save_path

    def _get_csv_fieldnames(self):
        """Define the CSV columns used by the experiment log."""
        return [
            "mode",
            "mask_source",
            "timestamp",
            "prompt",
            "num_inference_steps",
            "image_guidance_scale",
            "guidance_scale",
            "image_size",
            "input_save_path",
            "output_save_path",
        ]

    def _migrate_csv_if_needed(self):
        """Upgrade old CSV files to the current header when needed."""
        fieldnames = self._get_csv_fieldnames()
        if not self.log_csv_path.exists():
            return

        with self.log_csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            existing_fieldnames = reader.fieldnames or []
            if existing_fieldnames == fieldnames:
                return

            rows = list(reader)

        with self.log_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "mode": row.get("mode", "global_edit"),
                        "mask_source": row.get("mask_source", "not_applicable"),
                        "timestamp": row.get("timestamp", ""),
                        "prompt": row.get("prompt", ""),
                        "num_inference_steps": row.get("num_inference_steps", ""),
                        "image_guidance_scale": row.get("image_guidance_scale", ""),
                        "guidance_scale": row.get("guidance_scale", ""),
                        "image_size": row.get("image_size", ""),
                        "input_save_path": row.get("input_save_path", ""),
                        "output_save_path": row.get("output_save_path", ""),
                    }
                )

    def _build_summary_text(
        self,
        mode: str,
        mask_source: str,
        prompt: str,
        num_inference_steps: int,
        image_guidance_scale: float,
        guidance_scale: float,
        input_save_path: Path,
        output_save_path: Path,
    ) -> str:
        """Build a readable multi-line summary for the current experiment."""
        if mode == "local_inpaint":
            image_guidance_line = "图像引导强度: 该模式未使用"
        else:
            image_guidance_line = f"图像引导强度: {float(image_guidance_scale)}"

        return (
            "实验信息摘要\n"
            f"编辑模式: {self.get_mode_label(mode)}\n"
            f"Mask来源: {self.get_mask_source_label(mask_source)}\n"
            f"模型名称: {self.get_model_name(mode)}\n"
            f"运行设备: {self.get_device_name(mode)}\n"
            f"编辑指令: {prompt.strip()}\n"
            f"推理步数: {int(num_inference_steps)}\n"
            f"{image_guidance_line}\n"
            f"文本引导强度: {float(guidance_scale)}\n"
            f"图像尺寸: {self.image_size[0]} x {self.image_size[1]}\n"
            f"输入图保存路径: {input_save_path}\n"
            f"输出图保存路径: {output_save_path}"
        )

    def _append_experiment_log(
        self,
        mode: str,
        mask_source: str,
        timestamp: str,
        prompt: str,
        num_inference_steps: int,
        image_guidance_scale: float,
        guidance_scale: float,
        input_save_path: Path,
        output_save_path: Path,
    ) -> None:
        """Append one experiment record to docs/experiment_log.csv."""
        fieldnames = self._get_csv_fieldnames()
        row = {
            "mode": mode,
            "mask_source": mask_source,
            "timestamp": timestamp,
            "prompt": prompt.strip(),
            "num_inference_steps": int(num_inference_steps),
            "image_guidance_scale": float(image_guidance_scale),
            "guidance_scale": float(guidance_scale),
            "image_size": f"{self.image_size[0]}x{self.image_size[1]}",
            "input_save_path": str(input_save_path),
            "output_save_path": str(output_save_path),
        }

        self._migrate_csv_if_needed()
        file_exists = self.log_csv_path.exists()
        with self.log_csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def edit_image(
        self,
        input_image: Image.Image,
        prompt: str,
        mode: str = "global_edit",
        mask_source: str = "not_applicable",
        mask_image: Image.Image = None,
        drawn_mask_data=None,
        num_inference_steps: int = 20,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
    ) -> dict:
        """
        Run text-guided image editing with the loaded diffusion pipelines.
        """
        if input_image is None:
            raise ValueError("Input image cannot be None.")

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if mode != "local_inpaint":
            mask_source = "not_applicable"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image = self.preprocess_image(input_image)
        input_save_path = self._save_image(image, self.input_dir, "input")

        if mode == "local_inpaint":
            if mask_source == "drawn_mask":
                mask = self.extract_drawn_mask(drawn_mask_data)
            else:
                if mask_image is None:
                    raise ValueError("Mask image cannot be None in local_inpaint mode.")
                mask = self.preprocess_mask(mask_image)

            result = self.get_pipeline(mode)(
                prompt=prompt.strip(),
                image=image,
                mask_image=mask,
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
            ).images[0]
        else:
            result = self.get_pipeline(mode)(
                prompt=prompt.strip(),
                image=image,
                num_inference_steps=int(num_inference_steps),
                image_guidance_scale=float(image_guidance_scale),
                guidance_scale=float(guidance_scale),
            ).images[0]

        output_save_path = self._save_image(result, self.output_dir, "output")
        self._append_experiment_log(
            mode=mode,
            mask_source=mask_source,
            timestamp=timestamp,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            input_save_path=input_save_path,
            output_save_path=output_save_path,
        )
        summary_text = self._build_summary_text(
            mode=mode,
            mask_source=mask_source,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            input_save_path=input_save_path,
            output_save_path=output_save_path,
        )

        return {
            "result_image": result,
            "input_save_path": str(input_save_path),
            "output_save_path": str(output_save_path),
            "mode": mode,
            "mask_source": mask_source,
            "model_name": self.get_model_name(mode),
            "device": self.get_device_name(mode),
            "summary_text": summary_text,
        }
