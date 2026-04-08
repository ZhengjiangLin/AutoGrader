import mimetypes
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

import requests
from canvasapi import Canvas
from canvasapi.exceptions import InvalidAccessToken

from .config import Settings


@dataclass
class SubmissionFile:
    """Represents a downloaded student submission file."""
    name: str
    path: Path


class CanvasService:
    """Handles all communication with Canvas LMS (download submissions, submit grades)."""

    def __init__(self, settings: Settings):
        """Initialize connection to Canvas using token from settings."""
        if not settings.canvas_token:
            raise ValueError("Missing CANVAS_TOKEN. Please set it in your .env file.")

        self.settings = settings
        self.canvas = Canvas(settings.canvas_url, settings.canvas_token)

        try:
            self.course = self.canvas.get_course(settings.course_id)
            self.assignment = self.course.get_assignment(settings.assignment_id)
        except InvalidAccessToken as exc:
            raise ValueError(
                "Canvas token is invalid or expired. Please generate a new Access Token in Canvas Settings."
            ) from exc

    def list_submissions(self):
        """Return all submissions for the current assignment."""
        return self.assignment.get_submissions(include=["user"])

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Remove dangerous characters from filename to avoid errors."""
        sanitized = re.sub(r'[<>:"/\\|?*]+', "_", name).strip()
        return sanitized or "submission_file"

    @staticmethod
    def _natural_sort_key(value: str):
        """Sort files in natural order (1, 2, 10 instead of 1, 10, 2)."""
        return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]

    @staticmethod
    def _infer_ext_from_content_type(content_type: str | None) -> str:
        """Guess file extension from HTTP Content-Type header."""
        if not content_type:
            return ""
        pure_type = content_type.split(";", 1)[0].strip().lower()
        guessed = mimetypes.guess_extension(pure_type) or ""
        return ".jpg" if guessed == ".jpe" else guessed

    def _download_with_retry(self, file_url: str) -> requests.Response:
        """Download file with automatic retry on network issues."""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.get(file_url, timeout=90)
                resp.raise_for_status()
                return resp
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError):
                if attempt >= max_attempts:
                    raise
                time.sleep(1.5 * attempt)

        raise RuntimeError("Failed to download attachment after maximum retries.")

    @staticmethod
    def _parse_canvas_time(raw_value: str | None) -> datetime | None:
        """Convert Canvas time string to Python datetime object."""
        if not raw_value:
            return None
        value = raw_value.strip()
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _can_use_cached_file(
        self,
        save_path: Path,
        *,
        attachment_size: int | None,
        attachment_updated_at: str | None,
    ) -> bool:
        """Check if we already have the latest version of this file locally."""
        if not save_path.exists() or not save_path.is_file():
            return False

        if attachment_size is not None:
            try:
                if save_path.stat().st_size != int(attachment_size):
                    return False
            except Exception:
                return False

        updated_at = self._parse_canvas_time(attachment_updated_at)
        if updated_at is not None:
            file_mtime = datetime.fromtimestamp(save_path.stat().st_mtime, tz=timezone.utc)
            if file_mtime < updated_at:
                return False

        return True

    def download_attachments(self, submission, download_dir: Path) -> list[SubmissionFile]:
        """Download all attachments for one student submission."""
        files: list[SubmissionFile] = []
        attachments = getattr(submission, "attachments", []) or []
        attachments = sorted(
            attachments,
            key=lambda item: self._natural_sort_key(
                str(item.get("filename") if isinstance(item, dict) else getattr(item, "filename", ""))
            ),
        )

        for attachment in attachments:
            file_url = getattr(attachment, "url", None)
            filename = getattr(attachment, "filename", None)
            attachment_id = getattr(attachment, "id", None)
            attachment_size = getattr(attachment, "size", None)
            attachment_updated_at = getattr(attachment, "updated_at", None)

            if isinstance(attachment, dict):
                file_url = file_url or attachment.get("url")
                filename = filename or attachment.get("filename")
                attachment_id = attachment_id or attachment.get("id")
                attachment_size = attachment_size or attachment.get("size")
                attachment_updated_at = attachment_updated_at or attachment.get("updated_at")

            if not file_url or not filename:
                continue

            filename = self._sanitize_filename(filename)
            if attachment_id is not None:
                save_name = self._sanitize_filename(f"{submission.user['name']}_{attachment_id}_{filename}")
            else:
                save_name = self._sanitize_filename(f"{submission.user['name']}_{filename}")

            save_path = download_dir / save_name

            # If we already have the latest file, skip download
            if self._can_use_cached_file(
                save_path,
                attachment_size=attachment_size if isinstance(attachment_size, int) else None,
                attachment_updated_at=attachment_updated_at,
            ):
                files.append(SubmissionFile(name=save_path.name, path=save_path))
                continue

            response = self._download_with_retry(file_url)

            # Add correct file extension if missing
            if save_path.suffix == "":
                inferred = self._infer_ext_from_content_type(response.headers.get("Content-Type"))
                if inferred:
                    save_path = save_path.with_suffix(inferred)

            save_path.write_bytes(response.content)
            files.append(SubmissionFile(name=save_path.name, path=save_path))

        return files

    def submit_grade_and_comment(self, submission, total_score, comment: str | None = None) -> None:
        """Submit final grade and comment back to Canvas."""
        payload = {"submission": {"posted_grade": total_score}}
        if comment and comment.strip():
            payload["comment"] = {"text_comment": comment}
        submission.edit(**payload)