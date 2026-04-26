from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.status import Status

app = typer.Typer(
    name="workout-advisor",
    help="RAG-powered personal workout plan generator.",
    add_completion=False,
)
console = Console()


@app.command()
def generate(
    ankle_ok: bool = typer.Option(
        False,
        "--ankle-ok",
        help="Re-enable plyometrics (use when ankle has recovered).",
    ),
    export: bool = typer.Option(
        False,
        "--export",
        help="Save the generated plan to plans/YYYY-MM-DD.md.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Build and print the LLM prompt without calling Ollama (useful for debugging).",
    ),
) -> None:
    """Generate a personalised 7-day workout plan."""
    from planner import build_prompt, generate_plan, retrieve_week
    from query import get_personal_context

    console.print()
    console.print(
        Panel(
            "[bold cyan]Workout RAG Advisor[/bold cyan]\n"
            "[dim]Local RAG · LlamaIndex · Ollama qwen2.5:7b[/dim]",
            expand=False,
            border_style="cyan",
        )
    )
    console.print()

    exclude_plyo = not ankle_ok
    if exclude_plyo:
        console.print("[yellow]⚠  Plyometric mode: OFF  (ankle injury)[/yellow]")
    else:
        console.print("[green]✓  Plyometric mode: ON[/green]")
    console.print()

    # ── Retrieval ────────────────────────────────────────────────────────────
    with Status("[bold]Retrieving exercises from vector store...[/bold]", console=console):
        ctx = get_personal_context()
        week = retrieve_week(exclude_plyo=exclude_plyo)

    total = sum(len(v) for v in week.values())
    console.print(f"[green]✓[/green] Retrieved [bold]{total}[/bold] exercises across 7 days")

    # ── Dry run ──────────────────────────────────────────────────────────────
    if dry_run:
        prompt = build_prompt(week, ctx, exclude_plyo=exclude_plyo)
        console.print()
        console.print(Rule("[dim]LLM Prompt (dry run)[/dim]"))
        console.print(prompt)
        console.print(Rule())
        raise typer.Exit()

    # ── Generate ─────────────────────────────────────────────────────────────
    console.print()
    with Status(
        "[bold]Generating plan with Ollama (this takes 30–120 s)...[/bold]",
        console=console,
        spinner="dots",
    ):
        try:
            plan = generate_plan(exclude_plyo=exclude_plyo)
        except Exception as exc:
            console.print(f"\n[red]✗ Ollama error:[/red] {exc}")
            console.print(
                "[dim]Is Ollama running? Start it with: ollama serve "
                "and ensure qwen2.5:7b is pulled.[/dim]"
            )
            raise typer.Exit(code=1)

    # ── Display ──────────────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold cyan]Your 7-Day Plan[/bold cyan]"))
    console.print()
    console.print(Markdown(plan))
    console.print()

    # ── Export ───────────────────────────────────────────────────────────────
    if export:
        plans_dir = Path("plans")
        plans_dir.mkdir(exist_ok=True)
        out_path = plans_dir / f"{date.today().isoformat()}.md"
        out_path.write_text(
            f"# Workout Plan — {date.today().isoformat()}\n\n{plan}\n",
            encoding="utf-8",
        )
        console.print(f"[green]✓[/green] Plan saved to [bold]{out_path}[/bold]")

    console.print()


@app.command()
def query(
    search: str = typer.Argument(..., help="Exercise query, e.g. 'vertical jump explosive lower body'"),
    top_k: int = typer.Option(10, "--top-k", help="Number of results to return."),
    exclude_plyo: bool = typer.Option(False, "--exclude-plyo", help="Filter out plyometric exercises."),
) -> None:
    """Search the exercise database by semantic query."""
    from query import filter_exercises, node_to_exercise, retrieve_exercises

    console.print()
    with Status(f"[bold]Searching for:[/bold] {search}", console=console):
        nodes = retrieve_exercises(search, top_k=top_k * 2)
        filtered = filter_exercises(nodes, exclude_plyo=exclude_plyo)

    results = [node_to_exercise(n) for n in filtered[:top_k]]

    if not results:
        console.print("[yellow]No matching exercises found.[/yellow]")
        raise typer.Exit()

    console.print(f"\n[bold]Top {len(results)} results for:[/bold] [cyan]{search}[/cyan]\n")
    for i, ex in enumerate(results, 1):
        muscles = ", ".join(ex["primary_muscles"]) if ex["primary_muscles"] else "—"
        equip = ex["equipment"] or "body only"
        console.print(
            f"  [bold cyan]{i:2}.[/bold cyan] [bold]{ex['name']}[/bold]"
            f"  [dim]| {equip} | {ex['level']} | muscles: {muscles}[/dim]"
        )
    console.print()


if __name__ == "__main__":
    app()
