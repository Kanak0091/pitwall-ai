"""
Monte Carlo stochastic race simulation engine for PitWall AI.

Runs 1,000+ randomized race scenarios with varying Safety Cars, weather
showers, and lap-time variance to estimate win probability, podium probability,
and race-time distributions for competing strategies.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.simulator import RaceSimulator, RaceStrategy
from src.weather import build_weather_timeline
from src.safety_car import generate_safety_car_schedule


@dataclass
class MonteCarloOutput:
    """Container for aggregated Monte Carlo metrics and raw trial results."""
    summary_df: pd.DataFrame
    raw_times_df: pd.DataFrame
    n_simulations: int


class MonteCarloEngine:
    """Runs large-scale stochastic race simulations across candidate strategies."""

    def __init__(
        self,
        simulator: RaceSimulator,
        n_simulations: int = 1000,
        lap_noise_std: float = 0.22,
    ) -> None:
        if n_simulations < 10:
            raise ValueError("n_simulations should be at least 10.")
        self.simulator = simulator
        self.n_simulations = n_simulations
        self.lap_noise_std = lap_noise_std

    def run(
        self,
        strategies: list[RaceStrategy],
        initial_weather: str = "dry",
        rain_probability: float = 0.20,
        sc_probability: float = 0.45,
        seed: int = 42,
    ) -> MonteCarloOutput:
        """Execute `n_simulations` randomized Grand Prix runs across `strategies`."""
        if not strategies:
            raise ValueError("At least one RaceStrategy must be provided.")

        total_laps = self.simulator.total_laps
        original_noise = self.simulator.lap_noise_std
        self.simulator.lap_noise_std = self.lap_noise_std

        seed_seq = np.random.SeedSequence(seed)
        child_seeds = seed_seq.spawn(self.n_simulations)

        n_strats = len(strategies)
        times_matrix = np.zeros((self.n_simulations, n_strats), dtype=float)

        try:
            for sim_idx in range(self.n_simulations):
                rng = np.random.default_rng(child_seeds[sim_idx])
                weather_seed = int(rng.integers(0, 1_000_000_000))
                sc_seed = int(rng.integers(0, 1_000_000_000))

                weather_timeline = build_weather_timeline(
                    total_laps=total_laps,
                    initial_condition=initial_weather,
                    dynamic=rain_probability > 0.0,
                    rain_probability=rain_probability,
                    seed=weather_seed,
                )
                sc_schedule = generate_safety_car_schedule(
                    total_laps=total_laps,
                    sc_probability=sc_probability,
                    seed=sc_seed,
                )

                for s_idx, strategy in enumerate(strategies):
                    lap_seed = int(rng.integers(0, 1_000_000_000))
                    res = self.simulator.simulate(
                        strategy=strategy,
                        weather_timeline=weather_timeline,
                        sc_schedule=sc_schedule,
                        seed=lap_seed,
                        record_history=False,
                    )
                    times_matrix[sim_idx, s_idx] = res.total_time_s
        finally:
            self.simulator.lap_noise_std = original_noise

        order = np.argsort(times_matrix, axis=1)
        ranks = np.empty_like(order)
        rows_idx = np.arange(self.n_simulations)[:, None]
        ranks[rows_idx, order] = np.arange(n_strats)

        podium_cutoff = min(3, n_strats)
        strat_names = [s.name for s in strategies]

        summary_rows = []
        for s_idx, strat in enumerate(strategies):
            strat_times = times_matrix[:, s_idx]
            strat_ranks = ranks[:, s_idx]

            win_prob = float(np.mean(strat_ranks == 0) * 100.0)
            podium_prob = float(np.mean(strat_ranks < podium_cutoff) * 100.0)

            summary_rows.append(
                {
                    "Strategy": strat.name,
                    "Stops": strat.stops,
                    "Compounds": " -> ".join(strat.compounds),
                    "WinProb_%": round(win_prob, 1),
                    "PodiumProb_%": round(podium_prob, 1),
                    "MeanTime_s": round(float(np.mean(strat_times)), 2),
                    "StdDev_s": round(float(np.std(strat_times)), 2),
                    "P10_Best_s": round(float(np.percentile(strat_times, 10)), 2),
                    "P90_Worst_s": round(float(np.percentile(strat_times, 90)), 2),
                }
            )

        summary_df = (
            pd.DataFrame(summary_rows)
            .sort_values(by=["WinProb_%", "PodiumProb_%", "MeanTime_s"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
        summary_df.insert(0, "Rank", summary_df.index + 1)

        raw_times_df = pd.DataFrame(times_matrix, columns=strat_names)

        return MonteCarloOutput(
            summary_df=summary_df,
            raw_times_df=raw_times_df,
            n_simulations=self.n_simulations,
        )