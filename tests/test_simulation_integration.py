import pytest
from policysim.simulation import Simulation, run_multiple
from policysim.brain import AgentBrain


def test_simulation_runs_without_llm_backend():
    """The heuristic backend must always work with zero external
    dependencies -- this is what guarantees a live demo never breaks on stage."""
    sim = Simulation(brain=AgentBrain(backend="heuristic"), seed=1)
    stats = sim.run_n_rounds(10)
    assert len(stats) == 10
    df = sim.history_df()
    assert not df.isnull().values.any()
    assert (df["unemployment_rate"] >= 0).all() and (df["unemployment_rate"] <= 1).all()
    assert (df["gini"] >= 0).all() and (df["gini"] <= 1).all()


def test_policy_trigger_appears_in_event_log():
    sim = Simulation(brain=AgentBrain(backend="heuristic"), seed=2)
    sim.run_round()
    sim.apply_policy_event("cash_transfer", 50.0)
    assert len(sim.event_log) == 1
    assert sim.event_log[0]["policy_id"] == "cash_transfer"


def test_same_seed_is_deterministic():
    """Given the same seed and heuristic backend, two runs should produce
    identical histories -- important so demo rehearsals are reproducible.
    NOTE: each Simulation must be fully built AND run before the next one is
    constructed, since Simulation.__init__ reseeds the global `random` module
    -- constructing sim2 before running sim1 would advance the shared RNG
    state and make sim1's own rounds non-reproducible."""
    sim1 = Simulation(brain=AgentBrain(backend="heuristic"), seed=99)
    sim1.run_n_rounds(8)
    df1 = sim1.history_df()

    sim2 = Simulation(brain=AgentBrain(backend="heuristic"), seed=99)
    sim2.run_n_rounds(8)
    df2 = sim2.history_df()

    assert df1.equals(df2), "identical seeds should produce identical simulation histories"


def test_run_multiple_produces_one_run_id_per_run():
    df = run_multiple(num_runs=3, rounds_per_run=6, policy_event=("cash_transfer", 40.0), policy_round=3)
    assert df["run_id"].nunique() == 3
    assert set(df.groupby("run_id").size()) == {6}


def test_cash_transfer_increases_low_income_spending():
    """Basic face-validity check: a cash transfer to low/mid income households
    should raise their spending relative to a no-policy baseline."""
    baseline = Simulation(brain=AgentBrain(backend="heuristic"), seed=7)
    for r in range(1, 11):
        baseline.run_round()
    base_df = baseline.history_df()

    treated = Simulation(brain=AgentBrain(backend="heuristic"), seed=7)
    for r in range(1, 11):
        if r == 5:
            treated.apply_policy_event("cash_transfer", 200.0)
        treated.run_round()
    treated_df = treated.history_df()

    base_late = base_df[base_df["round"] > 5]["spend_low"].mean()
    treated_late = treated_df[treated_df["round"] > 5]["spend_low"].mean()
    assert treated_late > base_late
