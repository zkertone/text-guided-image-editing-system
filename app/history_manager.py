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
        return [
            [
                record["id"],
                record["created_at"],
                record["mode"],
                record["prompt"],
                record["status"],
            ]
            for record in self.get_recent_record_dicts(limit=limit)
        ]

    def get_recent_record_dicts(self, limit: int = 10) -> list[dict]:
        """Return recent non-deleted edit records as dictionaries."""
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

        return [dict(row) for row in rows]

    def get_recent_input_images(self, limit: int = 10) -> list[list]:
        """Return recent non-deleted input images for display in Gradio."""
        return [
            [
                image["id"],
                image["file_name"],
                image["created_at"],
                image["width"],
                image["height"],
            ]
            for image in self.get_recent_input_image_dicts(limit=limit)
        ]

    def get_recent_input_image_dicts(self, limit: int = 10) -> list[dict]:
        """Return recent non-deleted input images as dictionaries."""
        with get_connection() as connection:
            rows = connection.execute(
                """
                SELECT id, file_name, created_at, width, height
                FROM images
                WHERE user_id = ?
                    AND image_type = 'input'
                    AND is_deleted = 0
                ORDER BY id DESC
                LIMIT ?
                """,
                (self.user_id, int(limit)),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_record_detail(self, record_id: int) -> Optional[dict]:
        """Return one non-deleted edit record as a dictionary."""
        with get_connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM edit_records
                WHERE id = ?
                    AND user_id = ?
                    AND is_deleted = 0
                """,
                (int(record_id), self.user_id),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def get_record_images(self, record_id: int) -> dict:
        """Load images linked to one edit record."""
        detail = self.get_record_detail(record_id)
        images = {
            "input": None,
            "mask": None,
            "control": None,
            "output": None,
        }

        if detail is None:
            return images

        image_fields = {
            "input": detail.get("input_image_id"),
            "mask": detail.get("mask_image_id"),
            "control": detail.get("control_image_id"),
            "output": detail.get("output_image_id"),
        }

        for image_type, image_id in image_fields.items():
            if image_id is None:
                continue
            try:
                images[image_type] = self.load_image(image_id)
            except ValueError:
                images[image_type] = None

        return images

    def load_image(self, image_id: int) -> Image.Image:
        """Load an image BLOB from SQLite as a PIL image."""
        image_data = self.load_image_bytes(image_id)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image.load()
        return image

    def load_image_bytes(self, image_id: int) -> bytes:
        """Load image PNG bytes from SQLite."""
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

        return bytes(row["image_data"])

    def soft_delete_record(self, record_id: int) -> bool:
        """Mark one edit record as deleted without removing it physically."""
        with get_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE edit_records
                SET is_deleted = 1
                WHERE id = ?
                    AND user_id = ?
                    AND is_deleted = 0
                """,
                (int(record_id), self.user_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    def soft_delete_image(self, image_id: int) -> bool:
        """Mark one image as deleted without removing the BLOB physically."""
        with get_connection() as connection:
            cursor = connection.execute(
                """
                UPDATE images
                SET is_deleted = 1
                WHERE id = ?
                    AND user_id = ?
                    AND is_deleted = 0
                """,
                (int(image_id), self.user_id),
            )
            connection.commit()
            return cursor.rowcount > 0
