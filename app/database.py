import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "database" / "app.db"
DEFAULT_USER_ID = 1
DEFAULT_USERNAME = "demo_user"


def get_connection() -> sqlite3.Connection:
    """Create a SQLite connection for the project database."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database() -> None:
    """Create database tables and the default demo user."""
    with get_connection() as connection:
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_type TEXT NOT NULL,
                file_name TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                image_data BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS edit_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                mode TEXT NOT NULL,
                prompt TEXT NOT NULL,
                input_image_id INTEGER,
                mask_image_id INTEGER,
                control_image_id INTEGER,
                output_image_id INTEGER,
                mask_source TEXT,
                control_type TEXT,
                num_inference_steps INTEGER,
                image_guidance_scale REAL,
                guidance_scale REAL,
                status TEXT NOT NULL,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (input_image_id) REFERENCES images(id),
                FOREIGN KEY (mask_image_id) REFERENCES images(id),
                FOREIGN KEY (control_image_id) REFERENCES images(id),
                FOREIGN KEY (output_image_id) REFERENCES images(id)
            )
            """
        )

        cursor.execute(
            """
            INSERT OR IGNORE INTO users (id, username)
            VALUES (?, ?)
            """,
            (DEFAULT_USER_ID, DEFAULT_USERNAME),
        )

        connection.commit()
