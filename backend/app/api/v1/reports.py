"""
Report viewing API endpoints.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import List

router = APIRouter()


@router.get("/{discovery_id}")
async def list_reports(discovery_id: str):
    """
    List all reports for a discovery.

    Args:
        discovery_id: Discovery ID

    Returns:
        List of report metadata
    """
    try:
        reports_dir = Path(f"../outputs/{discovery_id}")

        if not reports_dir.exists():
            return {"reports": [], "count": 0}

        reports = []
        for report_file in reports_dir.glob("*.md"):
            stat = report_file.stat()
            reports.append({
                "report_id": report_file.stem,
                "filename": report_file.name,
                "size": stat.st_size,
                "created_at": stat.st_ctime,
                "modified_at": stat.st_mtime,
            })

        return {"reports": reports, "count": len(reports)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{discovery_id}/{report_id}")
async def get_report(discovery_id: str, report_id: str):
    """
    Get a specific report's content.

    Args:
        discovery_id: Discovery ID
        report_id: Report ID (filename without extension)

    Returns:
        Report content
    """
    try:
        report_path = Path(f"../outputs/{discovery_id}/{report_id}.md")

        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")

        with open(report_path, "r") as f:
            content = f.read()

        return {
            "report_id": report_id,
            "discovery_id": discovery_id,
            "content": content,
            "format": "markdown",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
