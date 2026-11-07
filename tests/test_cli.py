from typer.testing import CliRunner

from bem_helmholtz_3d.cli import app

runner = CliRunner()


def test_help():
    """The help message includes the CLI name."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Add the arguments and print the result" in result.stdout


def test_add_command():
    """The CLI prints the sum of its arguments."""
    result = runner.invoke(app, ["1", "2"], color=False)
    assert result.exit_code == 0
    assert "3" in result.stdout
