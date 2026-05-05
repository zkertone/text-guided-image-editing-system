from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL import Image

from app.database import DEFAULT_USER_ID, get_connection, init_database


class HistoryManager:
    """Store images and edit records in the local SQLite database."""

    def __init__(self, user_id: int = DEFAULT_USER_ID):
        init_database()
        self.user_id = user_id

    def save_image(
        self,
        image: Image.Image,
        image_type: str,
        file_name: Optional[str] = None,
    ) -> int:
        """Save a PIL image as PNG bytes in the images table."""
        image_rgb = image.convert("RGB")
        buffer = BytesIO()
        image_rgb.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        if file_name is None:
            file_name = f"{image_type}.png"
        else:
            file_name = Path(file_name).name

        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO images (
                    user_id,
                    image_type,
                    file_name,
                    mime_type,
                    width,
                    height,
                    image_data
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.user_id,
                    image_type,
                    file_name,
                    "image/png",
                    image_rgb.width,
                    image_rgb.height,
                    image_data,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def save_edit_record(
        self,
        mode: str,
        prompt: str,
        input_image_id: Optional[int],
        output_image_id: Optional[int],
        mask_image_id: Optional[int] = None,
        control_image_id: Optional[int] = None,
        mask_source: str = "not_applicable",
        control_type: str = "not_applicable",
        num_inference_steps: int = 20,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
        status: str = "success",
        error_message: Optional[str] = None,
    ) -> int:
        """Save one edit task record and related image IDs."""
        with get_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO edit_records (
                    user_id,
                    mode,
                    prompt,
                    input_image_id,
                    mask_image_id,
                    control_image_id,
                    output_image_id,
                    mask_source,
                    control_type,
                    num_inference_steps,
                    image_guidance_scale,
                    guidance_scale,
                    status,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.user_id,
                    mode,
                    prompt.strip(),
                    input_image_id,
                    mask_image_id,
                    control_image_id,
                    output_image_id,
                    mask_source,
                    control_type,
                    int(num_inference_steps),
                    float(image_guidance_scale),
                    float(guidance_scale),
                    status,
                    error_message,
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)

    def get_recent_records(self, limit: int = 10) -> list[list]:
        """Return recent edit records for display in Gradio."""
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, mode, prompt, status
                FROM edit_records
                WHERE user_id = ? AND is_deleted = 0
                ORDER BY id DESC
                LIMIT ?
                """,
                (self.user_id, int(limit)),
            ).fetchall()

        return [
            [
                row["id"],
                row["created_at"],
                row["mode"],
                row["prompt"],
                row["status"],
            ]
            for row in rows
        ]

    def load_image(self, image_id: int) -> Image.Image:
        """Load an image BLOB from SQLite as a PIL image."""
        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT image_data
                FROM images
                WHERE id = ? AND is_deleted = 0
                """,
                (int(image_id),),
            ).fetchone()

        if row is None:
            raise ValueError(f"Image not found: {image_id}")

        return Image.open(BytesIO(row["image_data"])).convert("RGB")
