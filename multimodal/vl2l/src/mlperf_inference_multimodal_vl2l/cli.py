"""The CLI definition for the VL2L benchmark."""

from __future__ import annotations

import typer

app = typer.Typer()


@app.command()
def main():
    """VL2L benchmark CLI"""
    typer.echo("Hello, World!")
