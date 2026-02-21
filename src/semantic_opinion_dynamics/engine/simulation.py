from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from semantic_opinion_dynamics import __version__
from semantic_opinion_dynamics.contracts import (
    AgentLog,
    NetworkSpec,
    OpinionState,
    PersonasSpec,
    RunConfig,
    RunMetadata,
    StepLog,
)
from semantic_opinion_dynamics.engine.agent import Agent
from semantic_opinion_dynamics.engine.network import Network
from semantic_opinion_dynamics.io.persistence import (
    init_run_dir,
    make_run_id,
    save_inputs_snapshot,
    save_run_metadata,
    save_step,
    utc_now_iso,
)
from semantic_opinion_dynamics.llm.factory import create_llm_client
from semantic_opinion_dynamics.llm.rate_limit import AsyncRateLimiter
from semantic_opinion_dynamics.llm.wrappers import RateLimitedLLMClient


class Simulation:
    def __init__(
        self,
        *,
        cfg: RunConfig,
        network: NetworkSpec,
        personas: PersonasSpec,
        config_path: str,
        network_path: str,
        personas_path: str,
    ) -> None:
        self.cfg = cfg
        self.network_spec = network
        self.personas_spec = personas
        self.config_path = config_path
        self.network_path = network_path
        self.personas_path = personas_path

        self.network = Network(network)
        self.agents: Dict[str, Agent] = {}

        missing = [nid for nid in self.network.node_ids() if nid not in personas.agents]
        if missing:
            raise ValueError(f"Missing personas for nodes: {missing}")

        for nid in self.network.node_ids():
            p = personas.agents[nid]
            self.agents[nid] = Agent(agent_id=nid, persona=p, state=p.initial_opinion)

    async def run(self) -> None:
        run_id = make_run_id(self.cfg.run.name)
        out_dir = Path(self.cfg.run.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = init_run_dir(out_dir, run_id)

        meta = RunMetadata(
            run_id=run_id,
            created_at_utc=utc_now_iso(),
            engine_version=__version__,
            config_path=self.config_path,
            network_path=self.network_path,
            personas_path=self.personas_path,
        )
        save_run_metadata(paths, meta)
        save_inputs_snapshot(
            paths,
            self.cfg,
            network_dict=self.network_spec.model_dump(mode="json"),
            personas_dict=self.personas_spec.model_dump(mode="json"),
        )

        base_llm = create_llm_client(self.cfg.llm)
        limiter = AsyncRateLimiter(
            max_concurrency=self.cfg.llm.max_concurrency,
            min_interval_sec=self.cfg.llm.min_interval_sec,
        )
        llm = RateLimitedLLMClient(base_llm, limiter)

        def _neighbors_for_log(aid: str) -> list[str]:
            incoming = self.network.incoming_neighbors(aid)
            incoming = sorted(incoming, key=lambda x: x[1], reverse=True)

            k = self.cfg.experiment.max_neighbors_in_prompt
            if k >= 0:
                incoming = incoming[:k]

            return [nb_id for nb_id, _w in incoming]

        step0 = StepLog(
            t=0,
            agents={
                aid: AgentLog(
                    agent_id=aid,
                    display_name=a.display_name(),
                    state=a.state,
                    neighbors=_neighbors_for_log(aid),
                    llm_update=None,
                    prompt=None,
                    error=None,
                )
                for aid, a in self.agents.items()
            },
        )
        save_step(paths, step0)

        for t in range(self.cfg.experiment.steps):
            current_states: Dict[str, OpinionState] = {aid: a.state for aid, a in self.agents.items()}

            tasks = []
            agent_ids: List[str] = []

            for aid, agent in self.agents.items():
                incoming = self.network.incoming_neighbors(aid)
                neighbors: List[Tuple[str, str, OpinionState, float]] = []
                for nb_id, w in incoming:
                    nb_agent = self.agents[nb_id]
                    neighbors.append((nb_id, nb_agent.display_name(), current_states[nb_id], w))

                agent_ids.append(aid)
                tasks.append(
                    agent.step(
                        policy_question=self.cfg.experiment.policy_question,
                        neighbors=neighbors,
                        llm=llm,
                        behavior_rules=self.cfg.prompts.behavior_rules,
                        include_neighbor_opinion_text=self.cfg.experiment.include_neighbor_opinion_text,
                        max_neighbors_in_prompt=self.cfg.experiment.max_neighbors_in_prompt,
                        store_prompts=self.cfg.experiment.store_prompts,
                        max_retries=self.cfg.llm.max_retries,
                    )
                )

            results = await _gather_all(tasks)

            next_states: Dict[str, OpinionState] = {}
            logs: Dict[str, AgentLog] = {}
            for aid, (next_state, log) in zip(agent_ids, results):
                next_states[aid] = next_state
                logs[aid] = log

            for aid, a in self.agents.items():
                a.state = next_states[aid]

            step = StepLog(t=t + 1, agents=logs)
            save_step(paths, step)


async def _gather_all(tasks):
    import asyncio

    return await asyncio.gather(*tasks)
