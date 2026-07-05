"""
server.py
---------
Serves the PolicySim engine over HTTP and hosts the 3D Three.js front-end
(policysim/static/) in the same process, so the visualization and the real
simulation are one single running project instead of two separate things.

Run:
    uvicorn policysim.server:app --reload
Then open:
    http://localhost:8000

Endpoints:
    GET  /api/state              current snapshot (households, firms, government, latest stats)
    POST /api/reset?seed=123     start a fresh simulation (seed optional)
    POST /api/step               advance one round, returns the new snapshot
    POST /api/policy/{policy_id} toggle a policy (subsidy_cut, minimum_wage_increase,
                                  luxury_tax, cash_transfer) — same on/off behavior as the
                                  color-card demo: showing it twice reverts it
"""

from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .simulation import Simulation
from .brain import AgentBrain
from .policies import POLICY_LIBRARY
from .view_helpers import snapshot

app = FastAPI(title="PolicySim")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict = {"sim": None}


def _sim() -> Simulation:
    if _state["sim"] is None:
        _state["sim"] = Simulation(brain=AgentBrain(backend="heuristic"))
    return _state["sim"]


@app.get("/api/state")
def get_state():
    return snapshot(_sim())


@app.post("/api/reset")
def reset(seed: int | None = None):
    _state["sim"] = Simulation(brain=AgentBrain(backend="heuristic"), seed=seed)
    return snapshot(_sim())


@app.post("/api/step")
def step():
    _sim().run_round()
    return snapshot(_sim())


@app.post("/api/policy/{policy_id}")
def toggle_policy(policy_id: str):
    if policy_id not in POLICY_LIBRARY:
        raise HTTPException(404, f"Unknown policy '{policy_id}'. Known: {list(POLICY_LIBRARY)}")
    msg = _sim().apply_policy_event(policy_id)
    return {"message": msg, **snapshot(_sim())}


# Serve the 3D front-end as static files, mounted last so /api/* above takes priority.
_static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
