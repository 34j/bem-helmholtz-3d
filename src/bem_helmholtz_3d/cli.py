import typer
from rich import print as rich_print

from .main import add

app = typer.Typer()


@app.command()
def main(n1: int, n2: int) -> None:
    """Add the arguments and print the result."""
    rich_print(add(n1, n2))
