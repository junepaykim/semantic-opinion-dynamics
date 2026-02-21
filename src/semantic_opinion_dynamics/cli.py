import asyncio
import json
from pathlib import Path
from typing import Optional

import typer

from semantic_opinion_dynamics.contracts import OpinionUpdate
from semantic_opinion_dynamics.engine.simulation import Simulation
from semantic_opinion_dynamics.io.loaders import load_personas_yaml, load_run_config, load_network_json

app = typer.Typer(add_completion=False)


@app.command()
def run(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    network: Path = typer.Option(..., exists=True, dir_okay=False),
    personas: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Optional[Path] = typer.Option(None, dir_okay=True),
) -> None:
    cfg = load_run_config(config)
    if output_dir is not None:
        cfg.run.output_dir = str(output_dir)

    net = load_network_json(network)
    pers = load_personas_yaml(personas)

    sim = Simulation(
        cfg=cfg,
        network=net,
        personas=pers,
        config_path=str(config),
        network_path=str(network),
        personas_path=str(personas),
    )
    asyncio.run(sim.run())


@app.command()
def schema() -> None:
    print(json.dumps(OpinionUpdate.model_json_schema(), ensure_ascii=False, indent=2))


@app.command()
def validate(
    config: Path = typer.Option(..., exists=True, dir_okay=False),
    network: Path = typer.Option(..., exists=True, dir_okay=False),
    personas: Path = typer.Option(..., exists=True, dir_okay=False),
) -> None:
    load_run_config(config)
    load_network_json(network)
    load_personas_yaml(personas)
    typer.echo("OK")


if __name__ == "__main__":
    app()
