import csv
from datetime import datetime
from pathlib import Path

from PIL import Image


class ImageEditor:
    """A lightweight wrapper around the InstructPix2Pix pipeline."""

    def __init__(self, pipeline, image_size=(512, 512)):
        self.pipeline = pipeline
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

    def get_model_name(self) -> str:
        """Get the current pipeline model name for display."""
        if hasattr(self.pipeline, "config") and hasattr(self.pipeline.config, "_name_or_path"):
            return str(self.pipeline.config._name_or_path)
        return "unknown"

    def get_device_name(self) -> str:
        """Get the current execution device for display."""
        if hasattr(self.pipeline, "_execution_device"):
            return str(self.pipeline._execution_device)
        if hasattr(self.pipeline, "device"):
            return str(self.pipeline.device)
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

    def _build_summary_text(
        self,
        prompt: str,
        num_inference_steps: int,
        image_guidance_scale: float,
        guidance_scale: float,
        input_save_path: Path,
        output_save_path: Path,
    ) -> str:
        """Build a readable multi-line summary for the current experiment."""
        return (
            "实验信息摘要\n"
            f"模型名称: {self.get_model_name()}\n"
            f"运行设备: {self.get_device_name()}\n"
            f"编辑指令: {prompt.strip()}\n"
            f"推理步数: {int(num_inference_steps)}\n"
            f"图像引导强度: {float(image_guidance_scale)}\n"
            f"文本引导强度: {float(guidance_scale)}\n"
            f"图像尺寸: {self.image_size[0]} x {self.image_size[1]}\n"
            f"输入图保存路径: {input_save_path}\n"
            f"输出图保存路径: {output_save_path}"
        )

    def _append_experiment_log(
        self,
        timestamp: str,
        prompt: str,
        num_inference_steps: int,
        image_guidance_scale: float,
        guidance_scale: float,
        input_save_path: Path,
        output_save_path: Path,
    ) -> None:
        """Append one experiment record to docs/experiment_log.csv."""
        fieldnames = [
            "timestamp",
            "prompt",
            "num_inference_steps",
            "image_guidance_scale",
            "guidance_scale",
            "image_size",
            "input_save_path",
            "output_save_path",
        ]

        row = {
            "timestamp": timestamp,
            "prompt": prompt.strip(),
            "num_inference_steps": int(num_inference_steps),
            "image_guidance_scale": float(image_guidance_scale),
            "guidance_scale": float(guidance_scale),
            "image_size": f"{self.image_size[0]}x{self.image_size[1]}",
            "input_save_path": str(input_save_path),
            "output_save_path": str(output_save_path),
        }

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
        num_inference_steps: int = 20,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
    ) -> dict:
        """
        Run text-guided image editing with the loaded diffusion pipeline.
        """
        if input_image is None:
            raise ValueError("Input image cannot be None.")

        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image = self.preprocess_image(input_image)
        input_save_path = self._save_image(image, self.input_dir, "input")

        result = self.pipeline(
            prompt=prompt.strip(),
            image=image,
            num_inference_steps=int(num_inference_steps),
            image_guidance_scale=float(image_guidance_scale),
            guidance_scale=float(guidance_scale),
        ).images[0]

        output_save_path = self._save_image(result, self.output_dir, "output")
        self._append_experiment_log(
            timestamp=timestamp,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            image_guidance_scale=image_guidance_scale,
            guidance_scale=guidance_scale,
            input_save_path=input_save_path,
            output_save_path=output_save_path,
        )
        summary_text = self._build_summary_text(
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
            "model_name": self.get_model_name(),
            "device": self.get_device_name(),
            "summary_text": summary_text,
        }
