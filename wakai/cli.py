"""Command-line interface for wakai."""

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="wakai: age-dating host stars of transiting planets")

GUI_BACKEND = Path(__file__).resolve().parent.parent / "gui" / "backend" / "server.py"


@app.callback()
def main() -> None:
    """wakai: age-dating host stars of transiting planets."""


@app.command()
def gui(
    host: str = typer.Option("127.0.0.1", help="Host to bind the GUI server to"),
    port: int = typer.Option(5200, help="Port to serve the GUI on (default=5200)"),
    output_dir: str = typer.Option(
        str(Path.home() / "ql" / "wakai"),
        help="Directory where pipeline results are saved (default: ~/ql/wakai)",
    ),
    open_browser: bool = typer.Option(
        True, "--open-browser/--no-browser", help="Open the GUI in a browser automatically"
    ),
    reload: bool = typer.Option(
        False,
        "--reload/--no-reload",
        help=(
            "Auto-restart the backend when its Python files "
            "(gui/backend/server.py, routes_*.py, gaia_client.py) change, "
            "so route/logic edits take effect without a manual restart. "
            "Development only -- off by default. Static frontend files "
            "(gui/*.js/html/css) and the gui/backend/run_*.py subprocess "
            "wrappers are already served/executed fresh regardless."
        ),
    ),
) -> None:
    """Launch the wakai stellar age-dating GUI."""
    if not GUI_BACKEND.exists():
        typer.secho(f"GUI backend not found at {GUI_BACKEND}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    resolved_output = Path(output_dir).expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["WAKAI_HOST"] = host
    env["WAKAI_PORT"] = str(port)
    env["WAKAI_OUTPUT_DIR"] = str(resolved_output)
    if reload:
        env["WAKAI_RELOAD"] = "1"

    if open_browser:
        def _open_when_ready() -> None:
            time.sleep(1.5)
            webbrowser.open(f"http://{host}:{port}")

        threading.Thread(target=_open_when_ready, daemon=True).start()

    typer.echo(f"Starting wakai GUI on http://{host}:{port} ...")
    typer.echo(f"Saving pipeline results to {resolved_output}")
    try:
        subprocess.run([sys.executable, str(GUI_BACKEND)], env=env, check=True)
    except KeyboardInterrupt:
        typer.echo("\nShutting down...")


if __name__ == "__main__":
    app()
