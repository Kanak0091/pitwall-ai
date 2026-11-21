"""
Lap-by-lap F1 race simulation engine for PitWall AI.

Combines tyre degradation, fuel burn, pit-stop time loss, dynamic weather,
and Safety Car / VSC neutralizations to compute total race time and lap telemetry.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from src.tyre_model import TyreDegradationModel
from src.weather import (
    needs_weather_tyre_switch,
    recommended_wet_weather_compound,
    weather_lap_time_penalty,
)
from src.safety_car import (
    effective_pit_loss,
    evaluate_sc_pit_decision,
    neutralized_lap_time,
)


@dataclass
class RaceStrategy:
    """Defines a tyre compound sequence and planned pit-stop laps."""
    name: str
    compounds: list[str]
    pit_laps: list[int]

    def __post_init__(self) -> None:
        self.compounds = [c.upper() for c in self.compounds]
        if len(self.pit_laps) != len(self.compounds) - 1:
            raise ValueError(
                f"Strategy '{self.name}' has {len(self.compounds)} stints "
                f"but {len(self.pit_laps)} pit laps (expected {len(self.compounds) - 1})."
            )
        if sorted(self.pit_laps) != list(self.pit_laps):
            raise ValueError(f"Pit laps must be strictly ascending: {self.pit_laps}")

    @property
    def stops(self) -> int:
        return len(self.pit_laps)


@dataclass
class SimulationResult:
    """Output container for a completed race simulation."""
    strategy_name: str
    total_time_s: float
    stops_made: int
    actual_pit_laps: list[int]
    compounds_used: list[str]
    lap_history: pd.DataFrame
    sc_recommendations: list[dict] = field(default_factory=list)


class RaceSimulator:
    """Simulates a full Grand Prix lap by lap."""

    def __init__(
        self,
        tyre_model: TyreDegradationModel,
        total_laps: int = 57,
        base_pit_loss: float = 22.0,
        lap_noise_std: float = 0.0,
        opportunistic_sc: bool = True,
    ) -> None:
        self.tyre_model = tyre_model
        self.total_laps = total_laps
        self.base_pit_loss = base_pit_loss
        self.lap_noise_std = lap_noise_std
        self.opportunistic_sc = opportunistic_sc

    def simulate(
        self,
        strategy: RaceStrategy,
        weather_timeline: list[str] | None = None,
        sc_schedule: dict[int, str] | None = None,
        seed: int | None = None,
        record_history: bool = True,
    ) -> SimulationResult:
        """
        Execute a lap-by-lap race simulation for a given strategy.

        Args:
            strategy: Planned compound sequence and pit laps.
            weather_timeline: Per-lap weather list ('dry', 'damp', 'wet').
            sc_schedule: Mapping {lap: 'NONE' | 'VSC' | 'SC'}.
            seed: Optional random seed when lap_noise_std > 0.
            record_history: If False, skips building the lap-by-lap DataFrame
                            for ~15x faster grid search and Monte Carlo runs.

        Returns:
            SimulationResult containing total race time and optional lap trace.
        """
        rng = np.random.default_rng(seed) if self.lap_noise_std > 0.0 else None

        if weather_timeline is None:
            weather_timeline = ["dry"] * self.total_laps
        elif len(weather_timeline) < self.total_laps:
            weather_timeline = weather_timeline + [weather_timeline[-1]] * (
                self.total_laps - len(weather_timeline)
            )

        if sc_schedule is None:
            sc_schedule = {}

        stint_idx = 0
        current_compound = strategy.compounds[0]
        if needs_weather_tyre_switch(current_compound, weather_timeline[0]):
            current_compound = recommended_wet_weather_compound(
                weather_timeline[0], fallback_dry=current_compound
            )

        tyre_age = 1
        total_time = 0.0
        actual_pit_laps: list[int] = []
        compounds_used: list[str] = [current_compound]
        sc_recommendations: list[dict] = []
        records = []

        remaining_planned_pits = list(strategy.pit_laps)

        for lap in range(1, self.total_laps + 1):
            weather = weather_timeline[lap - 1]
            track_status = sc_schedule.get(lap, "NONE")
            pit_this_lap = False
            pit_reason = ""
            next_compound = current_compound

            # 1. Evaluate Safety Car / VSC tactical decision on the first lap of an SC
            if record_history and track_status in {"SC", "VSC"} and (
                lap == 1 or sc_schedule.get(lap - 1, "NONE") == "NONE"
            ):
                next_planned = remaining_planned_pits[0] if remaining_planned_pits else None
                rec = evaluate_sc_pit_decision(
                    current_lap=lap,
                    total_laps=self.total_laps,
                    tyre_age=tyre_age,
                    next_planned_pit_lap=next_planned,
                    base_pit_loss=self.base_pit_loss,
                    track_status=track_status,
                )
                rec["lap"] = lap
                rec["track_status"] = track_status
                sc_recommendations.append(rec)

            # 2. Check weather-forced tyre switch (highest priority)
            if needs_weather_tyre_switch(current_compound, weather) and lap < self.total_laps:
                pit_this_lap = True
                pit_reason = f"Weather ({weather})"
                planned_dry = strategy.compounds[min(stint_idx, len(strategy.compounds) - 1)]
                next_compound = recommended_wet_weather_compound(weather, fallback_dry=planned_dry)

            # 3. Check opportunistic Safety Car pit stop (within 5 laps of scheduled stop)
            elif (
                self.opportunistic_sc
                and track_status in {"SC", "VSC"}
                and remaining_planned_pits
                and 0 < (remaining_planned_pits[0] - lap) <= 5
                and tyre_age >= 6
            ):
                pit_this_lap = True
                pit_reason = f"Opportunistic {track_status}"
                remaining_planned_pits.pop(0)
                stint_idx = min(stint_idx + 1, len(strategy.compounds) - 1)
                next_compound = strategy.compounds[stint_idx]

            # 4. Check scheduled dry pit stop
            elif remaining_planned_pits and lap >= remaining_planned_pits[0]:
                pit_this_lap = True
                pit_reason = "Scheduled"
                remaining_planned_pits.pop(0)
                stint_idx = min(stint_idx + 1, len(strategy.compounds) - 1)
                desired = strategy.compounds[stint_idx]
                next_compound = (
                    recommended_wet_weather_compound(weather, fallback_dry=desired)
                    if weather != "dry"
                    else desired
                )

            # 5. Compute base lap time from tyre + fuel model
            compound_model = self.tyre_model.get_model(current_compound)
            green_pace = compound_model.predict_lap_time(
                tyre_age=tyre_age,
                lap_number=lap,
                total_laps=self.total_laps,
            )

            # Add weather grip penalty
            green_pace += weather_lap_time_penalty(current_compound, weather)

            # Add stochastic lap variance (for Monte Carlo mode)
            if rng is not None:
                green_pace += float(rng.normal(0.0, self.lap_noise_std))

            # Apply SC / VSC pace neutralization
            lap_time = neutralized_lap_time(green_pace, track_status)

            # Add pit-stop time loss if boxing at the end of this lap
            pit_loss_applied = 0.0
            if pit_this_lap:
                pit_loss_applied = effective_pit_loss(self.base_pit_loss, track_status)
                lap_time = round(lap_time + pit_loss_applied, 3)
                actual_pit_laps.append(lap)
                current_compound = next_compound
                compounds_used.append(current_compound)
                tyre_age = 1
            else:
                tyre_age += 1

            total_time += lap_time

            if record_history:
                records.append(
                    {
                        "Lap": lap,
                        "Compound": current_compound,
                        "TyreAge": tyre_age if not pit_this_lap else 1,
                        "Weather": weather,
                        "TrackStatus": track_status,
                        "PitStop": pit_this_lap,
                        "PitReason": pit_reason,
                        "PitLoss_s": pit_loss_applied,
                        "LapTime_s": round(lap_time, 3),
                        "CumulativeTime_s": round(total_time, 3),
                    }
                )

        lap_df = pd.DataFrame(records) if record_history else pd.DataFrame()

        return SimulationResult(
            strategy_name=strategy.name,
            total_time_s=round(total_time, 3),
            stops_made=len(actual_pit_laps),
            actual_pit_laps=actual_pit_laps,
            compounds_used=compounds_used,
            lap_history=lap_df,
            sc_recommendations=sc_recommendations,
        )