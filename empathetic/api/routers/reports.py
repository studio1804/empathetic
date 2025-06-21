"""Report generation endpoints."""
from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
import tempfile
import os

from ..services.reports import ReportService

router = APIRouter()
report_service = ReportService()


@router.get("/generate/{model}")
async def generate_report(
    model: str,
    format: str = "html",
    include_validation: bool = False
):
    """Generate a report for model test results."""
    if format not in ["html", "json", "markdown"]:
        raise HTTPException(status_code=400, detail="Invalid format")
    
    try:
        report_content = await report_service.generate_report(
            model=model,
            format=format,
            include_validation=include_validation
        )
        
        if format == "html":
            return HTMLResponse(content=report_content)
        elif format == "json":
            return report_content
        else:  # markdown
            return FileResponse(
                path=report_content,
                media_type="text/markdown",
                filename=f"{model}_report.md"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{model}")
async def download_report(
    model: str,
    format: str = "html"
):
    """Download a report file."""
    try:
        file_path = await report_service.generate_report_file(
            model=model,
            format=format
        )
        
        return FileResponse(
            path=file_path,
            media_type="application/octet-stream",
            filename=f"{model}_report.{format}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/{model}")
async def get_dashboard_data(model: str):
    """Get dashboard data for a model."""
    try:
        data = await report_service.get_dashboard_data(model)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))