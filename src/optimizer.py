"""
Race strategy generation and optimization module for PitWall AI.

Evaluates 1-stop, 2-stop, and 3-stop tyre strategies across valid pit-stop
laps, ranks them by predicted total race time, and identifies the optimal
pit window (lap range) for the winning strategy.
"""

from dataclasses import dataclass
from itertools import product
import pandas as pd

from src.simulator import RaceSimulator, RaceStrategy, SimulationResult

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


@dataclass
class StrategyEvaluation:
    """Summary metrics for a single evaluated race strategy."""
    rank: int
    strategy_name: str
    stops: int
    compounds: list[str]
    pit_laps: list[int]
    total_time_s: float
    delta_to_best_s: float


class StrategyOptimizer:
    """Searches and ranks 1-stop, 2-stop, and 3-stop race strategies."""

    def __init__(
        self,
        simulator: RaceSimulator,
        min_stint_laps: int = 6,
        step_size: int = 2,
    ) -> None:
        self.simulator = simulator
        self.min_stint_laps = min_stint_laps
        self.step_size = max(1, step_size)

    def _valid_compound_sequences(
        self,
        num_stints: int,
        starting_compound: str | None = None,
    ) -> list[list[str]]:
        """Generate all legal F1 dry compound sequences (at least 2 compounds)."""
        sequences: list[list[str]] = []
        start_filter = starting_compound.upper() if starting_compound else None

        for combo in product(DRY_COMPOUNDS, repeat=num_stints):
            if start_filter and combo[0] != start_filter:
                continue
            if len(set(combo)) < 2:
                continue
            sequences.append(list(combo))

        return sequences

    def _valid_pit_schedules(self, num_stops: int) -> list[list[int]]:
        """Generate ascending pit-lap combinations respecting `min_stint_laps`."""
        total_laps = self.simulator.total_laps
        min_lap = self.min_stint_laps
        max_lap = total_laps - self.min_stint_laps

        if num_stops == 1:
            return [[lap] for lap in range(min_lap, max_lap + 1, self.step_size)]

        if num_stops == 2:
            schedules = []
            for p1 in range(min_lap, max_lap - self.min_stint_laps + 1, self.step_size):
                for p2 in range(p1 + self.min_stint_laps, max_lap + 1, self.step_size):
                    schedules.append([p1, p2])
            return schedules

        if num_stops == 3:
            schedules = []
            step3 = max(self.step_size, 3)
            for p1 in range(min_lap, max_lap - 2 * self.min_stint_laps + 1, step3):
                for p2 in range(p1 + self.min_stint_laps, max_lap - self.min_stint_laps + 1, step3):
                    for p3 in range(p2 + self.min_stint_laps, max_lap + 1, step3):
                        schedules.append([p1, p2, p3])
            return schedules

        raise ValueError("num_stops must be 1, 2, or 3")

    def optimize(
        self,
        starting_compound: str | None = None,
        stop_counts: tuple[int, ...] = (1, 2, 3),
        weather_timeline: list[str] | None = None,
        sc_schedule: dict[int, str] | None = None,
    ) -> tuple[pd.DataFrame, SimulationResult]:
        """Evaluate all candidate strategies and return ranked table + #1 full result."""
        results: list[tuple[RaceStrategy, SimulationResult]] = []

        for stops in stop_counts:
            sequences = self._valid_compound_sequences(stops + 1, starting_compound)
            schedules = self._valid_pit_schedules(stops)

            for seq in sequences:
                short_code = "-".join(c[0] for c in seq)
                for pit_laps in schedules:
                    strat_name = f"{short_code} ({stops}-Stop | Laps {','.join(map(str, pit_laps))})"
                    strategy = RaceStrategy(
                        name=strat_name,
                        compounds=seq,
                        pit_laps=pit_laps,
                    )
                    sim_res = self.simulator.simulate(
                        strategy=strategy,
                        weather_timeline=weather_timeline,
                        sc_schedule=sc_schedule,
                        record_history=False,
                    )
                    results.append((strategy, sim_res))

        if not results:
            raise RuntimeError("No valid strategies generated. Check total_laps and min_stint_laps.")

        results.sort(key=lambda item: item[1].total_time_s)
        best_strategy, _ = results[0]

        # Re-run only the winning strategy with full lap-by-lap telemetry
        best_result = self.simulator.simulate(
            strategy=best_strategy,
            weather_timeline=weather_timeline,
            sc_schedule=sc_schedule,
            record_history=True,
        )
        best_time = best_result.total_time_s

        rows = []
        for idx, (strat, res) in enumerate(results, start=1):
            rows.append(
                {
                    "Rank": idx,
                    "Strategy": strat.name,
                    "Stops": strat.stops,
                    "Compounds": " -> ".join(strat.compounds),
                    "PlannedPitLaps": strat.pit_laps,
                    "TotalTime_s": res.total_time_s,
                    "DeltaToBest_s": round(res.total_time_s - best_time, 3),
                }
            )

        return pd.DataFrame(rows), best_result

    def best_per_stop_count(self, ranked_df: pd.DataFrame) -> pd.DataFrame:
        """Extract the fastest strategy for each stop count (1-stop vs 2-stop vs 3-stop)."""
        best_by_stops = (
            ranked_df.sort_values("TotalTime_s")
            .groupby("Stops", as_index=False)
            .first()
            .sort_values("TotalTime_s")
            .reset_index(drop=True)
        )
        best_by_stops["Rank"] = best_by_stops.index + 1
        return best_by_stops

    def compute_pit_window_heatmap(
        self,
        compounds: list[str],
        weather_timeline: list[str] | None = None,
        sc_schedule: dict[int, str] | None = None,
        tolerance_s: float = 0.6,
    ) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
        """Evaluate pit-stop lap combinations to find the optimal pit window."""
        num_stops = len(compounds) - 1
        min_lap = self.min_stint_laps
        max_lap = self.simulator.total_laps - self.min_stint_laps
        records = []

        if num_stops == 1:
            for p1 in range(min_lap, max_lap + 1):
                strat = RaceStrategy("WindowScan", compounds, [p1])
                res = self.simulator.simulate(
                    strat, weather_timeline, sc_schedule, record_history=False
                )
                records.append({"PitLap1": p1, "TotalTime_s": res.total_time_s})

            df = pd.DataFrame(records)
            best_t = df["TotalTime_s"].min()
            df["Delta_s"] = (df["TotalTime_s"] - best_t).round(3)
            in_window = df[df["Delta_s"] <= tolerance_s]["PitLap1"]
            windows = [(int(in_window.min()), int(in_window.max()))]
            return df, windows

        if num_stops == 2:
            for p1 in range(min_lap, max_lap - self.min_stint_laps + 1, 2):
                for p2 in range(p1 + self.min_stint_laps, max_lap + 1, 2):
                    strat = RaceStrategy("WindowScan", compounds, [p1, p2])
                    res = self.simulator.simulate(
                        strat, weather_timeline, sc_schedule, record_history=False
                    )
                    records.append({"PitLap1": p1, "PitLap2": p2, "TotalTime_s": res.total_time_s})

            df = pd.DataFrame(records)
            best_t = df["TotalTime_s"].min()
            df["Delta_s"] = (df["TotalTime_s"] - best_t).round(3)
            in_window = df[df["Delta_s"] <= tolerance_s]
            windows = [
                (int(in_window["PitLap1"].min()), int(in_window["PitLap1"].max())),
                (int(in_window["PitLap2"].min()), int(in_window["PitLap2"].max())),
            ]
            return df, windows

        raise ValueError("Pit window heatmap supports 1-stop (2 compounds) or 2-stop (3 compounds).")