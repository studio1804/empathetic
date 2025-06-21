"""
Output management utilities for the Empathetic framework.
Handles organization and cleanup of logs, reports, and results.
"""

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import glob

class OutputManager:
    """Manages all output files and directories for the Empathetic framework"""
    
    def __init__(self, base_dir: str = "outputs"):
        self.base_dir = Path(base_dir)
        self.logs_dir = self.base_dir / "logs"
        self.reports_dir = self.base_dir / "reports"
        self.results_dir = self.base_dir / "results"
        self.artifacts_dir = self.base_dir / "artifacts"
        self.temp_dir = self.base_dir / "temp"
        
        # Ensure all directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create all output directories if they don't exist"""
        for directory in [self.logs_dir, self.reports_dir, self.results_dir, 
                         self.artifacts_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_log_file(self, log_type: str = "main") -> Path:
        """Get path for log file with organized naming"""
        timestamp = datetime.now().strftime('%Y%m%d')
        if log_type == "main":
            return self.logs_dir / f"empathetic_{timestamp}.log"
        elif log_type == "test":
            timestamp_full = datetime.now().strftime('%Y%m%d_%H%M%S')
            return self.logs_dir / f"test_execution_{timestamp_full}.log"
        elif log_type == "api":
            return self.logs_dir / f"api_calls_{timestamp}.log"
        else:
            return self.logs_dir / f"{log_type}_{timestamp}.log"
    
    def get_report_file(self, model: str, suite: str, format: str = "html") -> Path:
        """Get path for report file with organized naming"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"report_{model}_{suite}_{timestamp}.{format}"
        return self.reports_dir / filename
    
    def get_results_file(self, model: str, test_type: str = "comprehensive") -> Path:
        """Get path for results file with organized naming"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"results_{model}_{test_type}_{timestamp}.json"
        return self.results_dir / filename
    
    def get_artifact_file(self, artifact_type: str, model: str) -> Path:
        """Get path for artifact file (charts, analysis, etc.)"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{artifact_type}_{model}_{timestamp}"
        return self.artifacts_dir / filename
    
    def get_temp_file(self, prefix: str = "temp") -> Path:
        """Get path for temporary file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{prefix}_{timestamp}.tmp"
        return self.temp_dir / filename
    
    def save_results(self, results: Dict[str, Any], model: str, 
                    test_type: str = "comprehensive") -> Path:
        """Save test results to organized location"""
        results_file = self.get_results_file(model, test_type)
        
        # Add metadata
        results_with_meta = {
            "metadata": {
                "model": model,
                "test_type": test_type,
                "timestamp": datetime.now().isoformat(),
                "framework_version": "0.1.0"
            },
            "results": results
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_with_meta, f, indent=2, default=str)
        
        return results_file
    
    def cleanup_old_files(self, days_old: int = 30):
        """Clean up old files based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        # Clean logs older than specified days
        self._cleanup_directory(self.logs_dir, cutoff_date, "*.log")
        
        # Clean results older than 90 days
        results_cutoff = datetime.now() - timedelta(days=90)
        self._cleanup_directory(self.results_dir, results_cutoff, "*.json")
        
        # Clean temp files
        self._cleanup_directory(self.temp_dir, cutoff_date, "*")
        
    def _cleanup_directory(self, directory: Path, cutoff_date: datetime, pattern: str):
        """Clean files in directory older than cutoff date"""
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    try:
                        file_path.unlink()
                        print(f"Cleaned up old file: {file_path}")
                    except OSError:
                        pass  # File might be in use
    
    def get_recent_results(self, model: str, limit: int = 10) -> List[Path]:
        """Get recent result files for a model"""
        pattern = f"results_{model}_*.json"
        result_files = list(self.results_dir.glob(pattern))
        
        # Sort by modification time, newest first
        result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return result_files[:limit]
    
    def get_recent_reports(self, model: str, format: str = "html", limit: int = 10) -> List[Path]:
        """Get recent report files for a model"""
        pattern = f"report_{model}_*.{format}"
        report_files = list(self.reports_dir.glob(pattern))
        
        # Sort by modification time, newest first
        report_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return report_files[:limit]
    
    def get_storage_usage(self) -> Dict[str, Dict[str, Any]]:
        """Get storage usage statistics for each directory"""
        usage = {}
        
        for name, directory in [
            ("logs", self.logs_dir),
            ("reports", self.reports_dir), 
            ("results", self.results_dir),
            ("artifacts", self.artifacts_dir),
            ("temp", self.temp_dir)
        ]:
            if directory.exists():
                total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
                file_count = len([f for f in directory.rglob('*') if f.is_file()])
                
                usage[name] = {
                    "size_bytes": total_size,
                    "size_mb": round(total_size / (1024 * 1024), 2),
                    "file_count": file_count,
                    "path": str(directory)
                }
            else:
                usage[name] = {
                    "size_bytes": 0,
                    "size_mb": 0,
                    "file_count": 0,
                    "path": str(directory)
                }
        
        return usage

# Global output manager instance
output_manager = OutputManager()

def get_output_manager() -> OutputManager:
    """Get the global output manager instance"""
    return output_manager

def cleanup_outputs(days_old: int = 30):
    """Convenience function to clean up old output files"""
    output_manager.cleanup_old_files(days_old)

def get_storage_stats() -> Dict[str, Dict[str, Any]]:
    """Convenience function to get storage usage statistics"""
    return output_manager.get_storage_usage()