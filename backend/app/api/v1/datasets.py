"""
Dataset management API endpoints.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List, Optional

from app.dependencies import FileServiceDep
from app.models.responses import SuccessResponse

router = APIRouter()


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    discovery_id: Optional[str] = Query(None, description="Associate with discovery"),
    service: FileServiceDep = None,
):
    """
    Upload a dataset file.

    Args:
        file: File to upload
        discovery_id: Optional discovery ID to associate
        service: File service

    Returns:
        File path and metadata
    """
    try:
        file_path = await service.upload_dataset(file, discovery_id)
        return {
            "success": True,
            "filename": file.filename,
            "file_path": file_path,
            "discovery_id": discovery_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def list_datasets(
    discovery_id: Optional[str] = Query(None, description="Filter by discovery"),
    service: FileServiceDep = None,
):
    """
    List uploaded datasets.

    Args:
        discovery_id: Optional discovery ID to filter
        service: File service

    Returns:
        List of file information
    """
    try:
        files = service.list_uploaded_files(discovery_id)
        return {"files": files, "count": len(files)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{filename}", response_model=SuccessResponse)
async def delete_dataset(
    filename: str,
    discovery_id: Optional[str] = Query(None, description="Discovery ID"),
    service: FileServiceDep = None,
):
    """
    Delete an uploaded dataset.

    Args:
        filename: Filename to delete
        discovery_id: Optional discovery ID
        service: File service

    Returns:
        Success response
    """
    try:
        deleted = service.delete_file(filename, discovery_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="File not found")

        return SuccessResponse(
            success=True,
            message=f"File {filename} deleted successfully",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
