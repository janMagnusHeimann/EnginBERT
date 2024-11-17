import typer
import subprocess
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="EnginBERT",
    help="CLI tool for training and " +
    "evaluating BERT models on engineering papers"
)

console = Console()

# Define default paths
DEFAULT_SCRIPTS_DIR = Path("scripts")
PATHS = {
    "data": DEFAULT_SCRIPTS_DIR /
    "data_processing/data_arxiv.py",
    "preprocess": DEFAULT_SCRIPTS_DIR /
    "data_processing/preprocess_data.py",
    "mlm": DEFAULT_SCRIPTS_DIR /
    "train/mlm_training.py",
    "classification": DEFAULT_SCRIPTS_DIR /
    "train/train_bert_sequence_classification.py",
    "embeddings": DEFAULT_SCRIPTS_DIR /
    "helpers/embedding_extraction.py",
    "clustering": DEFAULT_SCRIPTS_DIR /
    "evaluation_metrics/category_clustering.py",
    "citations": DEFAULT_SCRIPTS_DIR /
    "evaluation_metrics/citation_evaluation.py",
    "ir": DEFAULT_SCRIPTS_DIR /
    "evaluation_metrics/information_retrieval.py"
}


def run_script(script_path: Path) -> bool:
    """Run a Python script and return True if successful."""
    try:
        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            check=True
        )
        console.print(f"[green]✓[/green] {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Error running {script_path}:[/red]\n{e.stderr}")
        return False


@app.command()
def train(
    scripts_dir: str = typer.Option(
        str(DEFAULT_SCRIPTS_DIR),
        "--scripts-dir",
        "-d",
        help="Directory containing the scripts"
    ),
    skip_steps: Optional[list[str]] = typer.Option(
        None,
        "--skip",
        "-s",
        help="Steps to skip " +
        "(data, preprocess, mlm, classification, embeddings)"
    )
):
    """Train the EnginBERT model from scratch."""
    steps = {
        "data": "Collecting arXiv data",
        "preprocess": "Preprocessing data",
        "mlm": "Fine-tuning BERT with MLM",
        "classification": "Training sequence classification",
        "embeddings": "Extracting embeddings"
    }

    skip_steps = skip_steps or []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        for step, description in steps.items():
            if step in skip_steps:
                console.print(f"[yellow]Skipping {description}...[/yellow]")
                continue

            script_path = Path(
                scripts_dir) / PATHS[step].relative_to(DEFAULT_SCRIPTS_DIR)
            if not script_path.exists():
                console.print(f"[red]✗ {script_path} not found![/red]")
                raise typer.Exit(1)

            progress.add_task(description, total=None)
            if not run_script(script_path):
                raise typer.Exit(1)


@app.command()
def evaluate(
    scripts_dir: str = typer.Option(
        str(DEFAULT_SCRIPTS_DIR),
        "--scripts-dir",
        "-d",
        help="Directory containing the scripts"
    ),
    metrics: Optional[list[str]] = typer.Option(
        None,
        "--metrics",
        "-m",
        help="Specific metrics to evaluate (clustering, citations, ir)"
    )
):
    """Evaluate the trained EnginBERT model."""
    available_metrics = {
        "clustering": "Evaluating category clustering",
        "citations": "Evaluating citation retrieval",
        "ir": "Evaluating information retrieval"
    }

    metrics = metrics or list(available_metrics.keys())
    invalid_metrics = set(metrics) - set(available_metrics.keys())
    if invalid_metrics:
        console.print("[red]Invalid metrics: " +
                      f"{', '.join(invalid_metrics)}[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        for metric in metrics:
            script_path = Path(
                scripts_dir) / PATHS[metric].relative_to(DEFAULT_SCRIPTS_DIR)
            if not script_path.exists():
                console.print(f"[red]✗ {script_path} not found![/red]")
                raise typer.Exit(1)

            progress.add_task(available_metrics[metric], total=None)
            if not run_script(script_path):
                raise typer.Exit(1)


@app.command()
def run_all(
    scripts_dir: str = typer.Option(
        str(DEFAULT_SCRIPTS_DIR),
        "--scripts-dir",
        "-d",
        help="Directory containing the scripts"
    )
):
    """Run the complete EnginBERT pipeline (training and evaluation)."""
    try:
        train(scripts_dir=scripts_dir)
        evaluate(scripts_dir=scripts_dir)
    except typer.Exit as e:
        raise typer.Exit(code=e.exit_code)


def main():
    app()


if __name__ == "__main__":
    main()
