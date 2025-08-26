import typer

app = typer.Typer()


@app.command()
def main() -> None:
    """Add the arguments and print the result."""
