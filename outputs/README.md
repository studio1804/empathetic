# Outputs Directory

This directory contains all generated files from the Empathetic framework.

## Directory Structure

- **`logs/`** - Application logs, debug information, and test execution logs
- **`reports/`** - Generated HTML, JSON, and Markdown reports
- **`results/`** - Raw test results and evaluation data
- **`artifacts/`** - Generated charts, visualizations, and analysis files
- **`temp/`** - Temporary files and intermediate processing data

## File Organization

### Logs
- `empathetic_YYYYMMDD.log` - Daily application logs
- `test_execution_YYYYMMDD_HHMMSS.log` - Individual test run logs
- `api_calls_YYYYMMDD.log` - API interaction logs

### Reports
- `report_MODEL_SUITE_YYYYMMDD_HHMMSS.html` - Detailed HTML reports
- `report_MODEL_SUITE_YYYYMMDD_HHMMSS.json` - Machine-readable results
- `summary_YYYYMMDD.md` - Human-readable summaries

### Results
- `results_MODEL_YYYYMMDD_HHMMSS.json` - Raw test results
- `empathy_scores_MODEL_YYYYMMDD.csv` - Detailed empathy dimension scores
- `bias_analysis_MODEL_YYYYMMDD.json` - Adversarial test results

## Retention Policy

- **Logs**: Kept for 30 days (automated cleanup)
- **Reports**: Kept indefinitely (user responsibility)
- **Results**: Kept for 90 days (automated cleanup)
- **Temp**: Cleaned on application restart

## Access

All files in this directory are git-ignored for privacy and size reasons.
Use the CLI commands or API to access recent results programmatically.