"""
Unit tests for PitWall AI core modules.

Run with:
    pytest -v
"""

import pandas as pd
import pytest

from src.data_loader import generate_synthetic_race_data
from src.tyre_model import TyreDegradationModel
from src.weather import (
    build_weather_timeline,
    needs_weather_tyre_switch,
    weather_lap_time_penalty,
)
from src.safety_car import (
    effective_pit_loss,
    evaluate_sc_pit_decision,
    generate_safety_car_schedule,
)
from src.simulator import RaceSimulator, RaceStrategy
from src.optimizer import StrategyOptimizer
from src.monte_carlo import MonteCarloEngine


@pytest.fixture
def fitted_tyre_model() -> TyreDegradationModel:
    """Fixture providing a trained TyreDegradationModel on synthetic data."""
    laps = generate_synthetic_race_data(total_laps=57, seed=42)
    return TyreDegradationModel(model_type="linear").fit(laps)


def test_synthetic_data_generation() -> None:
    df = generate_synthetic_race_data(total_laps=57, seed=42)
    assert not df.empty
    expected_cols = {"Driver", "LapNumber", "LapTimeSeconds", "Compound", "TyreLife", "TotalLaps"}
    assert expected_cols.issubset(set(df.columns))
    assert set(df["Compound"].unique()) == {"SOFT", "MEDIUM", "HARD"}


def test_tyre_degradation_ordering(fitted_tyre_model: TyreDegradationModel) -> None:
    soft = fitted_tyre_model.get_model("SOFT")
    med = fitted_tyre_model.get_model("MEDIUM")
    hard = fitted_tyre_model.get_model("HARD")

    # Soft should be fastest when fresh, Hard slowest when fresh
    assert soft.base_pace < med.base_pace < hard.base_pace
    # Soft should degrade faster per lap than Medium and Hard
    assert soft.deg_per_lap > med.deg_per_lap > hard.deg_per_lap


def test_fuel_correction_increases_pace_with_lighter_fuel(
    fitted_tyre_model: TyreDegradationModel,
) -> None:
    med = fitted_tyre_model.get_model("MEDIUM")
    lap1_time = med.predict_lap_time(tyre_age=1, lap_number=1, total_laps=57)
    lap50_time = med.predict_lap_time(tyre_age=1, lap_number=50, total_laps=57)
    assert lap50_time < lap1_time


def test_weather_tyre_switches_and_penalties() -> None:
    assert needs_weather_tyre_switch("MEDIUM", "damp") is True
    assert needs_weather_tyre_switch("INTERMEDIATE", "damp") is False
    assert needs_weather_tyre_switch("INTERMEDIATE", "dry") is True
    assert weather_lap_time_penalty("SOFT", "wet") > weather_lap_time_penalty("WET", "wet")


def test_safety_car_pit_discount_and_advice() -> None:
    assert effective_pit_loss(22.0, "NONE") == 22.0
    assert effective_pit_loss(22.0, "VSC") < 22.0
    assert effective_pit_loss(22.0, "SC") < effective_pit_loss(22.0, "VSC")

    advice = evaluate_sc_pit_decision(
        current_lap=21,
        total_laps=57,
        tyre_age=21,
        next_planned_pit_lap=23,
        base_pit_loss=22.0,
        track_status="SC",
    )
    assert advice["decision"] == "PIT NOW"


def test_race_simulator_execution(fitted_tyre_model: TyreDegradationModel) -> None:
    sim = RaceSimulator(fitted_tyre_model, total_laps=57, base_pit_loss=22.0)
    strategy = RaceStrategy("M-H", ["MEDIUM", "HARD"], [24])
    result = sim.simulate(strategy)

    assert result.stops_made == 1
    assert result.actual_pit_laps == [24]
    assert len(result.lap_history) == 57
    assert result.total_time_s > 5000.0


def test_strategy_validation_rejects_mismatched_pits() -> None:
    with pytest.raises(ValueError):
        RaceStrategy("Invalid", ["MEDIUM", "HARD"], [15, 35])


def test_optimizer_ranks_strategies(fitted_tyre_model: TyreDegradationModel) -> None:
    sim = RaceSimulator(fitted_tyre_model, total_laps=57)
    opt = StrategyOptimizer(sim, min_stint_laps=8, step_size=4)
    ranked_df, best_res = opt.optimize(starting_compound="MEDIUM", stop_counts=(1, 2))

    assert not ranked_df.empty
    assert ranked_df.iloc[0]["DeltaToBest_s"] == 0.0
    assert ranked_df["TotalTime_s"].is_monotonic_increasing
    assert best_res.total_time_s == ranked_df.iloc[0]["TotalTime_s"]


def test_monte_carlo_probabilities(fitted_tyre_model: TyreDegradationModel) -> None:
    sim = RaceSimulator(fitted_tyre_model, total_laps=57)
    strategies = [
        RaceStrategy("M-H (1-Stop)", ["MEDIUM", "HARD"], [24]),
        RaceStrategy("S-H-M (2-Stop)", ["SOFT", "HARD", "MEDIUM"], [15, 37]),
    ]
    mc = MonteCarloEngine(sim, n_simulations=50)
    out = mc.run(strategies, rain_probability=0.2, sc_probability=0.4, seed=123)

    assert len(out.summary_df) == 2
    assert pytest.approx(out.summary_df["WinProb_%"].sum(), abs=0.5) == 100.0