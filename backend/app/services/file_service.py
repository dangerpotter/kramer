"""
File service for handling uploads and downloads.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile
from datetime import datetime


class FileService:
    """Service for managing file uploads and downloads."""

    def __init__(self):
        """Initialize file service."""
        self.upload_dir = Path("../data/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def upload_dataset(
        self,
        file: UploadFile,
        discovery_id: Optional[str] = None,
    ) -> str:
        """
        Upload a dataset file.

        Args:
            file: Uploaded file
            discovery_id: Optional discovery ID to associate

        Returns:
            File path
        """
        # Create subdirectory for discovery if provided
        if discovery_id:
            target_dir = self.upload_dir / discovery_id
        else:
            target_dir = self.upload_dir

        target_dir.mkdir(parents=True, exist_ok=True)

        # Save file
        file_path = target_dir / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        return str(file_path.absolute())

    def list_uploaded_files(self, discovery_id: Optional[str] = None) -> List[dict]:
        """
        List uploaded files.

        Args:
            discovery_id: Optional discovery ID to filter

        Returns:
            List of file information
        """
        if discovery_id:
            target_dir = self.upload_dir / discovery_id
        else:
            target_dir = self.upload_dir

        if not target_dir.exists():
            return []

        files = []
        for file_path in target_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "path": str(file_path.absolute()),
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })

        return files

    def get_file_path(self, filename: str, discovery_id: Optional[str] = None) -> Optional[Path]:
        """
        Get path to an uploaded file.

        Args:
            filename: Filename
            discovery_id: Optional discovery ID

        Returns:
            File path or None if not found
        """
        if discovery_id:
            file_path = self.upload_dir / discovery_id / filename
        else:
            file_path = self.upload_dir / filename

        if file_path.exists() and file_path.is_file():
            return file_path

        return None

    def delete_file(self, filename: str, discovery_id: Optional[str] = None) -> bool:
        """
        Delete an uploaded file.

        Args:
            filename: Filename
            discovery_id: Optional discovery ID

        Returns:
            True if deleted, False if not found
        """
        file_path = self.get_file_path(filename, discovery_id)
        if file_path:
            file_path.unlink()
            return True
        return False
