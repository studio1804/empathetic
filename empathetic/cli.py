import click
import asyncio
import os
import sys
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from .config import config

console = Console()

@click.group()
@click.version_option(version='0.1.0')
def main():
    """Empathetic - AI Testing for Human Values"""
    pass

@main.command()
@click.argument('model')
@click.option('--suite', '-s', multiple=True, 
              help='Test suites: bias, alignment, fairness, safety, empathy, employment, healthcare')
@click.option('--output', '-o', default='terminal', 
              help='Output format: terminal, json, html, streamlit')
@click.option('--threshold', '-t', default=0.9, type=float,
              help='Minimum passing score')
@click.option('--adversarial', is_flag=True,
              help='Enable adversarial testing for bias detection')
@click.option('--dimensions', is_flag=True,
              help='Show detailed empathy dimension scores')
@click.option('--verbose', '-v', is_flag=True, 
              help='Verbose output')
def test(model: str, suite: tuple, output: str, threshold: float, adversarial: bool, dimensions: bool, verbose: bool):
    """Run comprehensive tests on a model"""
    console.print(f"[bold blue]Testing {model}...[/bold blue]")
    
    try:
        # Import here to avoid circular imports
        from .core.tester import Tester
        from .reports.generator import ReportGenerator
        
        tester = Tester()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running tests...", total=None)
            # Build config from CLI options
            config = {
                'adversarial': adversarial,
                'dimensions': dimensions,
                'threshold': threshold
            }
            
            results = asyncio.run(tester.run_tests(
                model, 
                suites=list(suite) or ['all'],
                config=config,
                verbose=verbose
            ))
        
        if output == 'terminal':
            display_results(results, verbose)
        elif output == 'streamlit':
            try:
                import streamlit
            except ImportError:
                console.print("[red]Error: Streamlit not installed.[/red]")
                console.print("Install with: [cyan]poetry install --extras streamlit[/cyan]")
                console.print("Or: [cyan]pip install streamlit plotly[/cyan]")
                sys.exit(1)
            launch_streamlit_report(results, model)
        else:
            generator = ReportGenerator()
            generator.create(results, format=output)
        
        if results.overall_score < threshold:
            console.print(f"[red]Failed: Score {results.overall_score:.3f} < {threshold}[/red]")
            sys.exit(1)
        
        console.print(f"[green]Passed: Score {results.overall_score:.3f}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)

@main.command()
@click.argument('model')
@click.option('--suite', '-s', required=True,
              help='Specific suite to check')
@click.option('--quick', '-q', is_flag=True,
              help='Quick check with subset of tests')
def check(model: str, suite: str, quick: bool):
    """Quick check against specific suite"""
    console.print(f"[bold blue]Quick check: {model} against {suite}[/bold blue]")
    
    try:
        from .core.tester import Tester
        
        tester = Tester()
        results = asyncio.run(tester.run_tests(
            model, 
            suites=[suite],
            quick=quick
        ))
        
        if suite in results.suite_results:
            result = results.suite_results[suite]
            console.print(f"Suite: {suite}")
            console.print(f"Score: {result.score:.3f}")
            console.print(f"Tests: {result.tests_passed}/{result.tests_total}")
        else:
            console.print(f"[red]No results for suite: {suite}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise click.Exit(1)

@main.command()
@click.argument('model')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed capability information')
def capabilities(model: str, verbose: bool):
    """Detect and display model capabilities"""
    console.print(f"[bold blue]Detecting capabilities for {model}...[/bold blue]")
    
    try:
        from .core.tester import Tester
        
        tester = Tester()
        provider = tester._get_provider(model)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Detecting capabilities...", total=None)
            capabilities = asyncio.run(provider.detect_capabilities())
            
        display_capabilities(capabilities, verbose)
        
        # Show adaptive testing recommendations
        from .models.capabilities import AdaptiveTester
        adaptive_tester = AdaptiveTester(capabilities)
        recommendations = adaptive_tester.get_testing_recommendations()
        
        if recommendations:
            console.print("\n[bold cyan]Adaptive Testing Recommendations:[/bold cyan]")
            for area, recommendation in recommendations.items():
                console.print(f"• [yellow]{area.title()}:[/yellow] {recommendation}")
            
            # Show recommended test suites
            recommended_suites = adaptive_tester.get_recommended_test_suites()
            console.print(f"\n[bold]Recommended test suites:[/bold] {', '.join(recommended_suites)}")
            
            # Show if adversarial testing is recommended
            if adaptive_tester.should_use_adversarial_testing():
                console.print("[bold red]⚠ Adversarial testing recommended due to bias susceptibility[/bold red]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise click.Exit(1)

@main.command()
@click.argument('path')
@click.option('--recursive', '-r', is_flag=True)
@click.option('--threshold', '-t', default=0.9, type=float)
def validate(path: str, recursive: bool, threshold: float):
    """Validate models meet standards"""
    console.print(f"[bold blue]Validating models at {path}[/bold blue]")
    console.print("[yellow]Validation command not yet implemented[/yellow]")

@main.command()
@click.option('--format', '-f', default='html',
              help='Report format: html, json, markdown')
@click.option('--input', '-i', help='Results file to report on')
def report(format: str, input: Optional[str]):
    """Generate detailed reports"""
    console.print(f"[bold blue]Generating {format} report[/bold blue]")
    
    try:
        from .reports.generator import ReportGenerator
        
        generator = ReportGenerator()
        if input:
            generator.create_from_file(input, format=format)
        else:
            console.print("[yellow]No input file specified[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        raise click.Exit(1)

@main.command()
@click.option('--force', '-f', is_flag=True, help='Overwrite existing .env file')
def setup(force: bool):
    """Interactive setup to create .env file with API keys"""
    console.print("[bold blue]Empathetic Setup[/bold blue]")
    console.print("This will help you configure API keys for AI providers.\n")
    
    env_file = Path('.env')
    
    # Check if .env already exists
    if env_file.exists() and not force:
        if not Confirm.ask(f"[yellow].env file already exists. Overwrite it?[/yellow]"):
            console.print("Setup cancelled.")
            return
    
    # Copy .env.example if .env doesn't exist
    env_example = Path('.env.example')
    if not env_file.exists() and env_example.exists():
        import shutil
        shutil.copy(env_example, env_file)
        console.print("[dim]Created .env from .env.example[/dim]")
    
    # OpenAI setup
    console.print("[bold cyan]OpenAI Configuration[/bold cyan]")
    console.print("Get your API key from: https://platform.openai.com/api-keys")
    
    current_openai = config.openai_api_key
    openai_key = Prompt.ask(
        "Enter your OpenAI API key",
        password=True,
        default="[current]" if current_openai else ""
    )
    
    if openai_key and openai_key != "[current]":
        if not openai_key.startswith("sk-"):
            console.print("[yellow]Warning: OpenAI API keys typically start with 'sk-'[/yellow]")
        config.save_api_key("openai", openai_key)
    
    # Anthropic setup
    console.print("\n[bold cyan]Anthropic Configuration (Optional)[/bold cyan]")
    console.print("Get your API key from: https://console.anthropic.com/")
    
    if Confirm.ask("Do you want to configure Anthropic API?", default=False):
        current_anthropic = config.anthropic_api_key
        anthropic_key = Prompt.ask(
            "Enter your Anthropic API key",
            password=True,
            default="[current]" if current_anthropic else ""
        )
        
        if anthropic_key and anthropic_key != "[current]":
            if not anthropic_key.startswith("sk-ant-"):
                console.print("[yellow]Warning: Anthropic API keys typically start with 'sk-ant-'[/yellow]")
            config.save_api_key("anthropic", anthropic_key)
    
    # HuggingFace setup
    console.print("\n[bold cyan]HuggingFace Configuration (Optional)[/bold cyan]")
    console.print("Get your API key from: https://huggingface.co/settings/tokens")
    
    if Confirm.ask("Do you want to configure HuggingFace API?", default=False):
        current_hf = config.huggingface_api_key
        hf_key = Prompt.ask(
            "Enter your HuggingFace API token",
            password=True,
            default="[current]" if current_hf else ""
        )
        
        if hf_key and hf_key != "[current]":
            config.save_api_key("huggingface", hf_key)
    
    # Verify configuration
    if config.has_api_keys():
        console.print("\n[green]✓ Configuration complete![/green]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print("1. Run environment check: [cyan]emp env-check[/cyan]")
        console.print("2. Run your first test: [cyan]emp test gpt-3.5-turbo[/cyan]")
        console.print("3. Start the API server: [cyan]emp serve[/cyan]")
    else:
        console.print("\n[yellow]Warning: No API keys configured.[/yellow]")
        console.print("You need at least one API key to run tests.")

@main.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=8000, type=int, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host: str, port: int, reload: bool):
    """Launch the Empathetic API server"""
    console.print(f"[bold blue]Starting Empathetic API server...[/bold blue]")
    
    # Check if API keys are configured
    if not config.has_api_keys():
        console.print(
            "[yellow]Warning: No API keys found. Run 'emp setup' to configure.[/yellow]"
        )
    
    console.print(f"[green]Server running at http://{host}:{port}[/green]")
    console.print("[dim]Press CTRL+C to stop[/dim]")
    
    try:
        import uvicorn
        uvicorn.run(
            "empathetic.api.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except ImportError:
        console.print("[red]Error: FastAPI dependencies not installed.[/red]")
        console.print("Run: pip install fastapi uvicorn")
        raise click.Exit(1)

@main.command()
def env_check():
    """Check environment configuration"""
    console.print("[bold blue]Environment Check[/bold blue]\n")
    
    checks = [
        ("OpenAI API Key", config.openai_api_key, "openai"),
        ("Anthropic API Key", config.anthropic_api_key, "anthropic"),
        ("HuggingFace API Key", config.huggingface_api_key, "huggingface"),
        ("Default Model", config.default_model, "default_model"),
        ("Config File", config.config_path, "config_path"),
    ]
    
    table = Table(title="Configuration Status")
    table.add_column("Setting", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Value", style="dim")
    
    all_good = True
    
    for name, value, key_type in checks:
        if value:
            # Mask sensitive values
            if "Key" in name or "Token" in name:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            status = "[green]✓ Set[/green]"
        else:
            display_value = "[dim]Not set[/dim]"
            if key_type in ["openai", "anthropic"]:
                status = "[yellow]○ Optional[/yellow]"
            else:
                status = "[dim]○ Optional[/dim]"
        
        table.add_row(name, status, display_value)
    
    # Check if at least one API key is set
    all_good = config.has_api_keys()
    
    console.print(table)
    
    # Check for .env file
    env_file = Path('.env')
    if env_file.exists():
        console.print(f"\n[green]✓ .env file found[/green]")
    else:
        console.print(f"\n[yellow]○ No .env file found[/yellow]")
        console.print("Run [cyan]emp setup[/cyan] to create one.")
    
    # Check config file
    if Path(config.config_path).exists():
        console.print(f"[green]✓ Config file found: {config.config_path}[/green]")
    else:
        console.print("[yellow]○ No config file found, using defaults[/yellow]")
    
    # Overall status
    if all_good:
        console.print("\n[bold green]✓ Ready to run tests![/bold green]")
        console.print("Try: [cyan]emp test gpt-3.5-turbo[/cyan]")
    else:
        console.print("\n[bold yellow]⚠ Setup required[/bold yellow]")
        console.print("Run: [cyan]emp setup[/cyan]")

def display_results(results, verbose: bool = False):
    """Display test results in terminal"""
    table = Table(title="Test Results Summary")
    
    table.add_column("Suite", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_column("Tests", style="green")
    table.add_column("Status", style="bold")
    
    for suite_name, result in results.suite_results.items():
        status = "✓ PASS" if result.score >= 0.9 else "⚠ WARN" if result.score >= 0.7 else "✗ FAIL"
        status_style = "green" if result.score >= 0.9 else "yellow" if result.score >= 0.7 else "red"
        
        table.add_row(
            suite_name,
            f"{result.score:.3f}",
            f"{result.tests_passed}/{result.tests_total}",
            f"[{status_style}]{status}[/{status_style}]"
        )
    
    console.print()
    console.print(table)
    console.print()
    console.print(f"[bold]Overall Score: {results.overall_score:.3f}[/bold]")
    
    if verbose and hasattr(results, 'recommendations'):
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in results.recommendations:
            console.print(f"• {rec}")

def display_capabilities(caps, verbose: bool = False):
    """Display model capabilities"""
    
    # Basic capabilities table
    basic_table = Table(title="Basic Capabilities")
    basic_table.add_column("Capability", style="cyan")
    basic_table.add_column("Value", style="green")
    
    basic_table.add_row("Context Length", f"{caps.context_length:,} tokens")
    basic_table.add_row("System Prompt", "✅ Yes" if caps.supports_system_prompt else "❌ No")
    basic_table.add_row("JSON Output", "✅ Yes" if caps.supports_json_output else "❌ No")
    basic_table.add_row("Function Calling", "✅ Yes" if caps.supports_function_calling else "❌ No")
    basic_table.add_row("Inference Speed", f"{caps.inference_speed:.1f} tokens/sec")
    basic_table.add_row("Average Latency", f"{caps.latency_avg:.2f} seconds")
    
    console.print(basic_table)
    
    # Empathy capabilities table
    empathy_table = Table(title="Empathy & Bias Capabilities")
    empathy_table.add_column("Metric", style="cyan")
    empathy_table.add_column("Score", style="magenta")
    empathy_table.add_column("Assessment", style="yellow")
    
    # Empathy baseline
    empathy_assessment = "🟢 Strong" if caps.empathy_baseline >= 0.8 else "🟡 Good" if caps.empathy_baseline >= 0.6 else "🔴 Weak"
    empathy_table.add_row("Empathy Baseline", f"{caps.empathy_baseline:.3f}", empathy_assessment)
    
    # Bias susceptibility (lower is better)
    bias_assessment = "🟢 Low Bias" if caps.bias_susceptibility <= 0.2 else "🟡 Moderate" if caps.bias_susceptibility <= 0.4 else "🔴 High Bias"
    empathy_table.add_row("Bias Susceptibility", f"{caps.bias_susceptibility:.3f}", bias_assessment)
    
    # Cultural awareness
    culture_assessment = "🟢 Strong" if caps.cultural_awareness >= 0.7 else "🟡 Good" if caps.cultural_awareness >= 0.5 else "🔴 Weak"
    empathy_table.add_row("Cultural Awareness", f"{caps.cultural_awareness:.3f}", culture_assessment)
    
    # Systemic thinking
    systemic_assessment = "🟢 Strong" if caps.systemic_thinking >= 0.6 else "🟡 Good" if caps.systemic_thinking >= 0.4 else "🔴 Weak"
    empathy_table.add_row("Systemic Thinking", f"{caps.systemic_thinking:.3f}", systemic_assessment)
    
    console.print(empathy_table)
    
    # Strengths and weaknesses
    if caps.strong_areas:
        console.print(f"\n[bold green]Strengths:[/bold green] {', '.join(caps.strong_areas)}")
    if caps.weak_areas:
        console.print(f"[bold red]Areas for Improvement:[/bold red] {', '.join(caps.weak_areas)}")
        
    # Detailed info if verbose
    if verbose:
        console.print(f"\n[bold]Detection Summary:[/bold]")
        console.print(f"• Total probe tests: {caps.total_probe_tests}")
        console.print(f"• Detection time: {caps.detection_time_seconds:.2f} seconds")
        console.print(f"• Error rate: {caps.error_rate:.1%}")
        console.print(f"• Consistency score: {caps.consistency_score:.3f}")
        
        console.print(f"\n[dim]Detected on: {caps.detection_date}[/dim]")

# Keys management commands
@main.group()
def keys():
    """Manage API keys"""
    pass

@keys.command(name='show')
def keys_show():
    """Show configured API keys (masked)"""
    table = Table(title="Configured API Keys")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Key (masked)", style="dim")
    
    providers = [
        ("OpenAI", config.openai_api_key),
        ("Anthropic", config.anthropic_api_key),
        ("HuggingFace", config.huggingface_api_key),
    ]
    
    for provider, key in providers:
        if key:
            masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
            status = "✓ Set"
        else:
            masked_key = "-"
            status = "Not set"
        
        table.add_row(provider, status, masked_key)
    
    console.print(table)
    
    if config.has_api_keys():
        console.print("\n[green]✓ At least one API key is configured[/green]")
    else:
        console.print("\n[yellow]⚠ No API keys configured. Run 'emp setup' to add keys.[/yellow]")

@keys.command(name='set')
@click.argument('provider', type=click.Choice(['openai', 'anthropic', 'huggingface']))
def keys_set(provider: str):
    """Set an API key for a provider"""
    console.print(f"[bold blue]Setting {provider.title()} API key[/bold blue]")
    
    # Show provider-specific help
    if provider == "openai":
        console.print("Get your API key from: https://platform.openai.com/api-keys")
        key_prefix = "sk-"
    elif provider == "anthropic":
        console.print("Get your API key from: https://console.anthropic.com/")
        key_prefix = "sk-ant-"
    else:  # huggingface
        console.print("Get your API key from: https://huggingface.co/settings/tokens")
        key_prefix = ""
    
    # Get current key
    current_key = config.get_api_key(provider)
    
    # Prompt for new key
    api_key = Prompt.ask(
        f"Enter your {provider.title()} API key",
        password=True,
        default="[cancel]" if not current_key else "[current]"
    )
    
    if api_key == "[cancel]":
        console.print("[yellow]Cancelled[/yellow]")
        return
    
    if api_key == "[current]":
        console.print("[dim]Keeping current key[/dim]")
        return
    
    # Validate key format
    if key_prefix and not api_key.startswith(key_prefix):
        console.print(f"[yellow]Warning: {provider.title()} API keys typically start with '{key_prefix}'[/yellow]")
        if not Confirm.ask("Continue anyway?"):
            console.print("[yellow]Cancelled[/yellow]")
            return
    
    # Save the key
    try:
        config.save_api_key(provider, api_key)
        console.print(f"[green]✓ {provider.title()} API key saved successfully[/green]")
        console.print("\nYou can now use this provider for testing.")
    except Exception as e:
        console.print(f"[red]Error saving API key: {e}[/red]")
        raise click.Exit(1)

@keys.command(name='remove')
@click.argument('provider', type=click.Choice(['openai', 'anthropic', 'huggingface']))
def keys_remove(provider: str):
    """Remove an API key for a provider"""
    current_key = config.get_api_key(provider)
    
    if not current_key:
        console.print(f"[yellow]No {provider.title()} API key is set[/yellow]")
        return
    
    if Confirm.ask(f"Remove {provider.title()} API key?"):
        try:
            config.save_api_key(provider, "")
            console.print(f"[green]✓ {provider.title()} API key removed[/green]")
        except Exception as e:
            console.print(f"[red]Error removing API key: {e}[/red]")
            raise click.Exit(1)
    else:
        console.print("[yellow]Cancelled[/yellow]")

@main.group()
def outputs():
    """Manage output files and directories"""
    pass

@outputs.command(name='info')
def outputs_info():
    """Show information about output directories and storage usage"""
    from .utils.outputs import get_storage_stats
    
    console.print("[bold blue]Output Directory Information[/bold blue]\n")
    
    stats = get_storage_stats()
    
    table = Table(title="Storage Usage")
    table.add_column("Directory", style="cyan")
    table.add_column("Files", justify="right")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Path", style="dim")
    
    total_size = 0
    total_files = 0
    
    for name, info in stats.items():
        table.add_row(
            name.title(),
            str(info['file_count']),
            str(info['size_mb']),
            info['path']
        )
        total_size += info['size_mb']
        total_files += info['file_count']
    
    table.add_row("", "", "", "", style="dim")
    table.add_row("TOTAL", str(total_files), f"{total_size:.2f}", "", style="bold")
    
    console.print(table)
    
    console.print(f"\n[dim]All output files are automatically git-ignored for privacy.[/dim]")

@outputs.command(name='clean')
@click.option('--days', '-d', default=30, type=int, 
              help='Clean files older than this many days')
@click.option('--dry-run', is_flag=True, 
              help='Show what would be deleted without actually deleting')
def outputs_clean(days: int, dry_run: bool):
    """Clean up old output files"""
    from .utils.outputs import cleanup_outputs
    
    if dry_run:
        console.print(f"[yellow]Dry run: Would clean files older than {days} days[/yellow]")
        # TODO: Implement dry run functionality
        console.print("[dim]Dry run not yet implemented - use without --dry-run to clean[/dim]")
    else:
        if Confirm.ask(f"Clean output files older than {days} days?"):
            try:
                cleanup_outputs(days)
                console.print(f"[green]✓ Cleaned output files older than {days} days[/green]")
            except Exception as e:
                console.print(f"[red]Error cleaning files: {e}[/red]")
        else:
            console.print("[yellow]Cancelled[/yellow]")

@outputs.command(name='recent')
@click.option('--model', '-m', help='Filter by model name')
@click.option('--type', '-t', type=click.Choice(['reports', 'results', 'logs']), 
              default='reports', help='Type of files to show')
@click.option('--limit', '-l', default=10, type=int, help='Number of files to show')
def outputs_recent(model: str, type: str, limit: int):
    """Show recent output files"""
    from .utils.outputs import get_output_manager
    
    output_manager = get_output_manager()
    
    console.print(f"[bold blue]Recent {type.title()}[/bold blue]")
    if model:
        console.print(f"[dim]Filtered by model: {model}[/dim]")
    console.print()
    
    if type == 'reports' and model:
        files = output_manager.get_recent_reports(model, limit=limit)
    elif type == 'results' and model:
        files = output_manager.get_recent_results(model, limit=limit)
    elif type == 'logs':
        # Show recent log files
        files = list(output_manager.logs_dir.glob("*.log"))
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        files = files[:limit]
    else:
        console.print("[yellow]Please specify --model for reports and results[/yellow]")
        return
    
    if not files:
        console.print(f"[dim]No recent {type} found[/dim]")
        return
    
    table = Table()
    table.add_column("File", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Modified", justify="right")
    
    for file_path in files:
        if file_path.exists():
            stat = file_path.stat()
            size_mb = stat.st_size / (1024 * 1024)
            modified = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            
            table.add_row(
                file_path.name,
                f"{size_mb:.2f} MB",
                modified
            )
    
    console.print(table)

def launch_streamlit_report(results, model: str):
    """Launch Streamlit report with test results"""
    try:
        from .utils.outputs import get_output_manager
        
        # Save results to temporary file for Streamlit to load
        output_manager = get_output_manager()
        temp_file = output_manager.get_temp_file("streamlit_data")
        
        # Convert results to dictionary if needed
        if hasattr(results, '__dict__'):
            results_dict = _results_to_dict(results)
        else:
            results_dict = results
            
        # Add model to results
        results_dict['model'] = model
        results_dict['timestamp'] = datetime.now().isoformat()
        
        # Save to temporary file
        with open(temp_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        console.print(f"[bold blue]Launching Streamlit report for {model}...[/bold blue]")
        console.print(f"[dim]Results saved to: {temp_file}[/dim]")
        
        # Launch Streamlit
        streamlit_file = Path(__file__).parent / "reports" / "streamlit_report.py"
        
        # Set environment variable for the data file
        env = os.environ.copy()
        env['EMPATHETIC_DATA_FILE'] = str(temp_file)
        
        # Launch Streamlit in a new process
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(streamlit_file),
            "--server.headless", "false",
            "--server.port", "8501"
        ], env=env)
        
    except ImportError:
        console.print("[red]Error: Streamlit not installed. Run: pip install streamlit[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error launching Streamlit: {str(e)}[/red]")
        sys.exit(1)

def _results_to_dict(results) -> dict:
    """Convert results object to dictionary"""
    if hasattr(results, '__dict__'):
        result_dict = {}
        for key, value in results.__dict__.items():
            if hasattr(value, '__dict__'):
                result_dict[key] = _results_to_dict(value)
            elif isinstance(value, dict):
                result_dict[key] = {k: _results_to_dict(v) if hasattr(v, '__dict__') else v 
                                   for k, v in value.items()}
            else:
                result_dict[key] = value
        return result_dict
    return results

if __name__ == "__main__":
    main()