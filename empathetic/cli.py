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
              help='Test suites: bias, alignment, fairness, safety, empathy, employment, healthcare')
@click.option('--output', '-o', default='terminal', 
              help='Output format: terminal, json, html')
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

if __name__ == "__main__":
    main()