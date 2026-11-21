"""
Weather modeling and tyre-suitability logic for PitWall AI.

Simulates dry, damp, and wet track conditions, calculates grip penalties
for mismatched tyres, and recommends weather-appropriate compounds.
"""

from dataclasses import dataclass
import numpy as np

VALID_WEATHER = {"dry", "damp", "wet"}
SLICK_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


@dataclass(frozen=True)
class WeatherState:
    """Represents track weather condition on a specific lap."""
    lap: int
    condition: str  # 'dry', 'damp', or 'wet'


def build_weather_timeline(
    total_laps: int,
    initial_condition: str = "dry",
    dynamic: bool = False,
    rain_probability: float = 0.0,
    seed: int | None = None,
) -> list[str]:
    """
    Generate a lap-by-lap list of weather conditions ('dry', 'damp', 'wet').

    Args:
        total_laps: Total number of laps in the race.
        initial_condition: Starting weather ('dry', 'damp', 'wet').
        dynamic: If True, weather can evolve between dry, damp, and wet.
        rain_probability: Chance (0.0 to 1.0) that a rain shower occurs mid-race.
        seed: Optional random seed for reproducibility.

    Returns:
        List of length `total_laps` with condition strings per lap.
    """
    condition = initial_condition.lower()
    if condition not in VALID_WEATHER:
        raise ValueError(f"Invalid weather condition: {initial_condition}")

    if not dynamic and rain_probability <= 0.0:
        return [condition] * total_laps

    rng = np.random.default_rng(seed)
    timeline = [condition] * total_laps

    # Trigger a realistic rain shower window if random draw succeeds
    if rng.random() < rain_probability and total_laps >= 15:
        start_lap = int(rng.integers(low=max(5, total_laps // 5), high=max(10, int(total_laps * 0.7))))
        duration = int(rng.integers(low=6, high=max(8, total_laps // 3)))
        heavy_rain = rng.random() < 0.45

        for lap_idx in range(start_lap, min(total_laps, start_lap + duration)):
            # Transition through damp -> wet (if heavy) -> damp -> dry
            if heavy_rain and (start_lap + 2 <= lap_idx <= start_lap + duration - 3):
                timeline[lap_idx] = "wet"
            else:
                timeline[lap_idx] = "damp"

    return timeline


def recommended_wet_weather_compound(condition: str, fallback_dry: str = "MEDIUM") -> str:
    """Return the ideal compound category for the current track state."""
    cond = condition.lower()
    if cond == "wet":
        return "WET"
    if cond == "damp":
        return "INTERMEDIATE"
    return fallback_dry.upper()


def needs_weather_tyre_switch(current_compound: str, condition: str) -> bool:
    """
    Check whether the current tyre is unsafe/uncompetitive for the weather.

    Rules:
    - 'dry' track requires slicks (SOFT/MEDIUM/HARD).
    - 'damp' track requires INTERMEDIATE.
    - 'wet' track requires WET (or INTERMEDIATE with a smaller penalty, but WET is optimal).
    """
    comp = current_compound.upper()
    cond = condition.lower()

    if cond == "dry" and comp not in SLICK_COMPOUNDS:
        return True
    if cond == "damp" and comp != "INTERMEDIATE":
        return True
    if cond == "wet" and comp != "WET":
        return True
    return False


def weather_lap_time_penalty(current_compound: str, condition: str) -> float:
    """
    Calculate extra seconds lost due to track dampness and tyre mismatch.

    Returns:
        Additional lap time in seconds (>= 0.0).
    """
    comp = current_compound.upper()
    cond = condition.lower()

    if cond == "dry":
        if comp in SLICK_COMPOUNDS:
            return 0.0
        # Wet/Inter tyres overheat and lose massive time on a dry track
        return 4.5 if comp == "INTERMEDIATE" else 8.0

    if cond == "damp":
        if comp == "INTERMEDIATE":
            return 3.5  # Baseline damp track slowness on optimal inters
        if comp == "WET":
            return 5.5
        return 14.0  # Slicks on a damp track lose huge time

    # cond == "wet"
    if comp == "WET":
        return 7.5  # Baseline heavy rain slowness on full wets
    if comp == "INTERMEDIATE":
        return 11.5
    return 24.0  # Slicks in heavy rain