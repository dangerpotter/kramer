"""
Report viewing and generation API endpoints.
"""

import os
import sys
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

# Add parent directory to path to import Kramer modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from app.core.kramer_bridge import get_bridge

router = APIRouter()


class GenerateReportRequest(BaseModel):
    """Request body for report generation."""
    report_type: str = Field(default="summary", description="Type of report: summary, detailed, executive")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    include_appendix: bool = Field(default=True, description="Include appendix with all findings")
    generate_narratives: bool = Field(default=True, description="Generate AI narratives (requires API key)")


class ReportMetadata(BaseModel):
    """Report metadata response."""
    report_id: str
    filename: str
    discovery_id: str
    report_type: str
    created_at: str
    file_path: str
    discoveries_count: int = 0
    total_findings: int = 0


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
                "id": report_file.stem,
                "name": report_file.stem.replace("_", " ").title(),
                "discovery_id": discovery_id,
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "file_path": str(report_file),
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

        with open(report_path, "r", encoding="utf-8") as f:
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


@router.post("/{discovery_id}/generate")
async def generate_report(
    discovery_id: str,
    request: GenerateReportRequest,
    background_tasks: BackgroundTasks,
):
    """
    Generate a new report for a discovery.

    Args:
        discovery_id: Discovery ID
        request: Report generation options

    Returns:
        Report metadata
    """
    try:
        # Get world model from bridge
        bridge = get_bridge()
        world_model = bridge.get_world_model(discovery_id)
        if not world_model:
            raise HTTPException(
                status_code=404,
                detail=f"Discovery {discovery_id} not found or world model not available"
            )

        # Import ReportGenerator
        from src.reporting.report_generator import ReportGenerator, ReportConfig

        # Create output directory
        output_dir = Path(f"../outputs/{discovery_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get API key for narrative generation
        api_key = os.getenv("ANTHROPIC_API_KEY") if request.generate_narratives else None

        # Create report generator
        generator = ReportGenerator(
            world_model=world_model,
            anthropic_api_key=api_key,
            min_confidence=request.min_confidence,
            max_discoveries=10,  # Allow more discoveries in report
        )

        # Generate report filename based on type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{request.report_type}_report_{timestamp}"
        report_path = output_dir / f"{report_name}.md"

        # Generate the report
        result = generator.generate_report(
            output_path=report_path,
            include_appendix=request.include_appendix,
            generate_narratives=request.generate_narratives and api_key is not None,
        )

        # Return metadata
        return {
            "report_id": report_name,
            "filename": f"{report_name}.md",
            "discovery_id": discovery_id,
            "report_type": request.report_type,
            "created_at": datetime.now().isoformat(),
            "file_path": str(report_path),
            "discoveries_count": result.get("discoveries_count", 0),
            "total_findings": result.get("total_findings", 0),
            "cost": result.get("cost", 0.0),
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{discovery_id}/{report_id}")
async def delete_report(discovery_id: str, report_id: str):
    """
    Delete a report.

    Args:
        discovery_id: Discovery ID
        report_id: Report ID

    Returns:
        Success message
    """
    try:
        report_path = Path(f"../outputs/{discovery_id}/{report_id}.md")

        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")

        report_path.unlink()

        return {"message": f"Report {report_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Cycle Report Endpoints

class CycleReportResponse(BaseModel):
    """Cycle report response model."""
    id: str
    cycle_id: str
    discovery_id: str
    summary: str
    tasks_completed: int
    findings_count: int
    hypotheses_count: int
    papers_count: int
    budget_used: float
    generation_cost: float
    created_at: str


@router.get("/{discovery_id}/cycle-reports")
async def list_cycle_reports(discovery_id: str):
    """
    List all cycle reports for a discovery.

    Args:
        discovery_id: Discovery ID

    Returns:
        List of cycle report summaries
    """
    try:
        from app.services.persistence_service import get_persistence_service

        persistence = get_persistence_service()
        reports = await persistence.get_cycle_reports(discovery_id)

        cycle_reports = []
        for report in reports:
            cycle_reports.append({
                "id": report.id,
                "cycle_id": report.cycle_id,
                "discovery_id": report.discovery_id,
                "summary": report.summary,
                "tasks_completed": report.tasks_completed,
                "findings_count": report.findings_count,
                "hypotheses_count": report.hypotheses_count,
                "papers_count": report.papers_count,
                "budget_used": report.budget_used,
                "generation_cost": report.generation_cost,
                "created_at": report.created_at.isoformat() if report.created_at else None,
            })

        return {"cycle_reports": cycle_reports, "count": len(cycle_reports)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{discovery_id}/cycle-reports/{cycle_id}")
async def get_cycle_report(discovery_id: str, cycle_id: str):
    """
    Get a specific cycle report's full content.

    Args:
        discovery_id: Discovery ID
        cycle_id: Cycle ID

    Returns:
        Full cycle report content
    """
    try:
        from app.services.persistence_service import get_persistence_service

        persistence = get_persistence_service()
        report = await persistence.get_cycle_report(cycle_id)

        if not report:
            raise HTTPException(status_code=404, detail="Cycle report not found")

        if report.discovery_id != discovery_id:
            raise HTTPException(status_code=404, detail="Cycle report not found for this discovery")

        return {
            "id": report.id,
            "cycle_id": report.cycle_id,
            "discovery_id": report.discovery_id,
            "summary": report.summary,
            "full_content": report.full_content,
            "tasks_completed": report.tasks_completed,
            "findings_count": report.findings_count,
            "hypotheses_count": report.hypotheses_count,
            "papers_count": report.papers_count,
            "budget_used": report.budget_used,
            "generation_cost": report.generation_cost,
            "created_at": report.created_at.isoformat() if report.created_at else None,
            "format": "markdown",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
