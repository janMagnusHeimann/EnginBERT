import typer
import subprocess
from typing import Optional, List
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from enum import Enum

app = typer.Typer(
    name="EnginBERT",
    help="CLI tool for training and evaluating domain-specific " +
    "BERT models on engineering papers"
)

console = Console()

# Define default paths
DEFAULT_SCRIPTS_DIR = Path("scripts")
PATHS = {
    # Data Processing
    "data": DEFAULT_SCRIPTS_DIR /
    "data_processing/data_arxiv.py",
    "preprocess": DEFAULT_SCRIPTS_DIR /
    "data_processing/preprocess_data.py",
    "knowledge_graph": DEFAULT_SCRIPTS_DIR /
    "data_processing/knowledge_graph/graph_builder.py",
    "visualize_kg": DEFAULT_SCRIPTS_DIR /
    "data_processing/knowledge_graph/visualize.py",

    # Training
    "mlm": DEFAULT_SCRIPTS_DIR /
    "train/mlm_training.py",
    "technical_term": DEFAULT_SCRIPTS_DIR /
    "train/technical_term_training.py",
    "equation": DEFAULT_SCRIPTS_DIR /
    "train/equation_understanding.py",
    "component": DEFAULT_SCRIPTS_DIR /
    "train/component_relation.py",
    "hierarchical": DEFAULT_SCRIPTS_DIR /
    "train/hierarchical_integration.py",

    # Evaluation
    "clustering": DEFAULT_SCRIPTS_DIR /
    "evaluation_metrics/category_clustering.py",
    "citations": DEFAULT_SCRIPTS_DIR /
    "evaluation_metrics/citation_evaluation.py",
    "ir": DEFAULT_SCRIPTS_DIR /
    "evaluation_metrics/information_retrieval.py"
}


class TrainingTask(str, Enum):
    MLM = "mlm"
    TECHNICAL_TERM = "technical_term"
    EQUATION = "equation"
    COMPONENT = "component"
    HIERARCHICAL = "hierarchical"
    ALL = "all"


class KGTask(str, Enum):
    BUILD = "build"
    VISUALIZE = "visualize"
    ALL = "all"


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
def knowledge_graph(
    task: KGTask = typer.Option(
        KGTask.ALL,
        "--task",
        "-t",
        help="Knowledge graph task to run"
    ),
    scripts_dir: str = typer.Option(
        str(DEFAULT_SCRIPTS_DIR),
        "--scripts-dir",
        "-d",
        help="Directory containing the scripts"
    )
):
    """Build and visualize the engineering knowledge graph."""
    tasks = {
        KGTask.BUILD: ("knowledge_graph", "Building knowledge graph"),
        KGTask.VISUALIZE: ("visualize_kg", "Visualizing knowledge graph")
    }

    if task == KGTask.ALL:
        selected_tasks = tasks
    else:
        selected_tasks = {task: tasks[task]}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        for script_key, description in selected_tasks.values():
            script_path = Path(
                scripts_dir) / PATHS[
                    script_key].relative_to(DEFAULT_SCRIPTS_DIR)
            if not script_path.exists():
                console.print(f"[red]✗ {script_path} not found![/red]")
                raise typer.Exit(1)

            progress.add_task(description, total=None)
            if not run_script(script_path):
                raise typer.Exit(1)


@app.command()
def train(
    scripts_dir: str = typer.Option(
        str(DEFAULT_SCRIPTS_DIR),
        "--scripts-dir",
        "-d",
        help="Directory containing the scripts"
    ),
    tasks: List[TrainingTask] = typer.Option(
        [TrainingTask.ALL],
        "--tasks",
        "-t",
        help="Training tasks to run"
    ),
    skip_data_prep: bool = typer.Option(
        False,
        "--skip-data",
        "-s",
        help="Skip data preparation"
    ),
    skip_kg: bool = typer.Option(
        False,
        "--skip-kg",
        "-k",
        help="Skip knowledge graph building"
    )
):
    """Train the EnginBERT model."""
    # Data preparation steps
    if not skip_data_prep:
        data_steps = {
            "data": "Collecting arXiv data",
            "preprocess": "Preprocessing data"
        }

        if not skip_kg:
            data_steps["knowledge_graph"] = "Building knowledge graph"

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for step, description in data_steps.items():
                script_path = Path(scripts_dir) / PATHS[
                    step].relative_to(DEFAULT_SCRIPTS_DIR)
                if not script_path.exists():
                    console.print(f"[red]✗ {script_path} not found![/red]")
                    raise typer.Exit(1)

                progress.add_task(description, total=None)
                if not run_script(script_path):
                    raise typer.Exit(1)

    # Training steps
    training_tasks = [TrainingTask.ALL] if TrainingTask.ALL in tasks else tasks
    if TrainingTask.ALL in training_tasks:
        training_tasks = [t for t in TrainingTask if t != TrainingTask.ALL]

    task_descriptions = {
        TrainingTask.MLM: "Running mlm pre-training",
        TrainingTask.TECHNICAL_TERM: "Running technical_term pre-training",
        TrainingTask.EQUATION: "Running equation understanding training",
        TrainingTask.COMPONENT: "Running component relation training",
        TrainingTask.HIERARCHICAL: "Running hierarchical integration"
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        for task in training_tasks:
            script_path = Path(scripts_dir) / PATHS[
                task].relative_to(DEFAULT_SCRIPTS_DIR)
            if not script_path.exists():
                console.print(f"[red]✗ {script_path} not found![/red]")
                raise typer.Exit(1)

            progress.add_task(task_descriptions[task], total=None)
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
        console.print("[red]Invalid metrics:" +
                      f" {', '.join(invalid_metrics)}[/red]")
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
    ),
    skip_data_prep: bool = typer.Option(
        False,
        "--skip-data",
        "-s",
        help="Skip data preparation"
    )
):
    """Run the complete EnginBERT pipeline (training and evaluation)."""
    try:
        train(scripts_dir=scripts_dir, tasks=[
            TrainingTask.ALL], skip_data_prep=skip_data_prep)
        evaluate(scripts_dir=scripts_dir)
    except typer.Exit as e:
        raise typer.Exit(code=e.exit_code)


def main():
    app()


if __name__ == "__main__":
    main()
