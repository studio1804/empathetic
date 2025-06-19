import click
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

console = Console()

@click.group()
@click.version_option(version='0.1.0')
def main():
    """Empathetic - AI Testing for Human Values"""
    pass

@main.command()
@click.argument('model')
@click.option('--suite', '-s', multiple=True, 
              help='Test suites: bias, alignment, fairness, safety')
@click.option('--output', '-o', default='terminal', 
              help='Output format: terminal, json, html')
@click.option('--threshold', '-t', default=0.9, type=float,
              help='Minimum passing score')
@click.option('--verbose', '-v', is_flag=True, 
              help='Verbose output')
def test(model: str, suite: tuple, output: str, threshold: float, verbose: bool):
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
            results = asyncio.run(tester.run_tests(
                model, 
                suites=list(suite) or ['all'],
                verbose=verbose
            ))
        
        if output == 'terminal':
            display_results(results, verbose)
        else:
            generator = ReportGenerator()
            generator.create(results, format=output)
        
        if results.overall_score < threshold:
            console.print(f"[red]Failed: Score {results.overall_score:.3f} < {threshold}[/red]")
            raise click.Exit(1)
        
        console.print(f"[green]Passed: Score {results.overall_score:.3f}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        if verbose:
            console.print_exception()
        raise click.Exit(1)

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
    
    env_vars = {}
    
    # OpenAI setup
    console.print("[bold cyan]OpenAI Configuration[/bold cyan]")
    console.print("Get your API key from: https://platform.openai.com/api-keys")
    
    openai_key = Prompt.ask(
        "Enter your OpenAI API key",
        password=True,
        default="" if not os.getenv("OPENAI_API_KEY") else "[current]"
    )
    
    if openai_key and openai_key != "[current]":
        if not openai_key.startswith("sk-"):
            console.print("[yellow]Warning: OpenAI API keys typically start with 'sk-'[/yellow]")
        env_vars["OPENAI_API_KEY"] = openai_key
    elif openai_key == "[current]" and os.getenv("OPENAI_API_KEY"):
        env_vars["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    
    # Anthropic setup
    console.print("\n[bold cyan]Anthropic Configuration (Optional)[/bold cyan]")
    console.print("Get your API key from: https://console.anthropic.com/")
    
    if Confirm.ask("Do you want to configure Anthropic API?", default=False):
        anthropic_key = Prompt.ask(
            "Enter your Anthropic API key",
            password=True,
            default="" if not os.getenv("ANTHROPIC_API_KEY") else "[current]"
        )
        
        if anthropic_key and anthropic_key != "[current]":
            if not anthropic_key.startswith("sk-ant-"):
                console.print("[yellow]Warning: Anthropic API keys typically start with 'sk-ant-'[/yellow]")
            env_vars["ANTHROPIC_API_KEY"] = anthropic_key
        elif anthropic_key == "[current]" and os.getenv("ANTHROPIC_API_KEY"):
            env_vars["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
    
    # HuggingFace setup
    console.print("\n[bold cyan]HuggingFace Configuration (Optional)[/bold cyan]")
    console.print("Get your API key from: https://huggingface.co/settings/tokens")
    
    if Confirm.ask("Do you want to configure HuggingFace API?", default=False):
        hf_key = Prompt.ask(
            "Enter your HuggingFace API token",
            password=True,
            default="" if not os.getenv("HUGGINGFACE_API_KEY") else "[current]"
        )
        
        if hf_key and hf_key != "[current]":
            env_vars["HUGGINGFACE_API_KEY"] = hf_key
        elif hf_key == "[current]" and os.getenv("HUGGINGFACE_API_KEY"):
            env_vars["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
    
    # Additional configuration
    console.print("\n[bold cyan]Additional Configuration[/bold cyan]")
    
    if Confirm.ask("Set default model?", default=False):
        default_model = Prompt.ask("Default model name", default="gpt-3.5-turbo")
        if default_model:
            env_vars["EMPATHETIC_DEFAULT_MODEL"] = default_model
    
    if Confirm.ask("Set custom config file path?", default=False):
        config_path = Prompt.ask("Config file path", default="./config/default.yaml")
        if config_path:
            env_vars["EMPATHETIC_CONFIG"] = config_path
    
    # Write .env file
    if env_vars:
        try:
            with open(env_file, 'w') as f:
                f.write("# Empathetic AI Testing Framework Configuration\n")
                f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
            
            console.print(f"\n[green]✓ Configuration saved to {env_file}[/green]")
            console.print("\n[bold]Next steps:[/bold]")
            console.print("1. Source the environment: [cyan]source .env[/cyan]")
            console.print("2. Or restart your terminal")
            console.print("3. Run your first test: [cyan]emp test gpt-3.5-turbo[/cyan]")
            
        except Exception as e:
            console.print(f"[red]Error writing .env file: {e}[/red]")
            raise click.Exit(1)
    else:
        console.print("[yellow]No configuration to save.[/yellow]")

@main.command()
def env_check():
    """Check environment configuration"""
    console.print("[bold blue]Environment Check[/bold blue]\n")
    
    checks = [
        ("OpenAI API Key", "OPENAI_API_KEY", True),
        ("Anthropic API Key", "ANTHROPIC_API_KEY", False),
        ("HuggingFace API Key", "HUGGINGFACE_API_KEY", False),
        ("Default Model", "EMPATHETIC_DEFAULT_MODEL", False),
        ("Config File", "EMPATHETIC_CONFIG", False),
    ]
    
    table = Table(title="Configuration Status")
    table.add_column("Setting", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Value", style="dim")
    
    all_good = True
    
    for name, env_var, required in checks:
        value = os.getenv(env_var)
        
        if value:
            # Mask sensitive values
            if "KEY" in env_var or "TOKEN" in env_var:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            status = "[green]✓ Set[/green]"
        else:
            display_value = "[dim]Not set[/dim]"
            if required:
                status = "[red]✗ Missing[/red]"
                all_good = False
            else:
                status = "[yellow]○ Optional[/yellow]"
        
        table.add_row(name, status, display_value)
    
    console.print(table)
    
    # Check for .env file
    env_file = Path('.env')
    if env_file.exists():
        console.print(f"\n[green]✓ .env file found[/green]")
    else:
        console.print(f"\n[yellow]○ No .env file found[/yellow]")
        console.print("Run [cyan]emp setup[/cyan] to create one.")
    
    # Check config file
    config_paths = [
        os.getenv('EMPATHETIC_CONFIG'),
        './empathetic.yaml',
        './config/default.yaml'
    ]
    
    config_found = False
    for path in config_paths:
        if path and Path(path).exists():
            console.print(f"[green]✓ Config file found: {path}[/green]")
            config_found = True
            break
    
    if not config_found:
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

if __name__ == "__main__":
    main()