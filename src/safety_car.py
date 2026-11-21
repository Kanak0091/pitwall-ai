"""
Safety Car (SC) and Virtual Safety Car (VSC) event modeling for PitWall AI.

Generates random or user-triggered neutralizations, adjusts pit-stop time loss,
and provides a 'Pit Now vs Stay Out' tactical recommendation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NeutralizationEvent:
    """Represents a Safety Car or VSC period."""
    start_lap: int
    end_lap: int
    event_type: str  # 'SC' or 'VSC'


def generate_safety_car_schedule(
    total_laps: int,
    sc_probability: float = 0.35,
    forced_sc_lap: int | None = None,
    seed: int | None = None,
) -> dict[int, str]:
    """
    Build a mapping of {lap_number: 'NONE' | 'VSC' | 'SC'}.

    Args:
        total_laps: Total race laps.
        sc_probability: Probability (0.0 to 1.0) of a random SC/VSC occurring.
        forced_sc_lap: Optional specific lap where the user triggers a Safety Car.
        seed: Optional random seed for reproducibility.

    Returns:
        Dictionary mapping each lap (1..total_laps) to track status.
    """
    schedule = {lap: "NONE" for lap in range(1, total_laps + 1)}
    rng = np.random.default_rng(seed)

    # Apply user-triggered Safety Car first (typically lasts 3 laps)
    if forced_sc_lap is not None and 1 <= forced_sc_lap <= total_laps:
        for lap in range(forced_sc_lap, min(total_laps + 1, forced_sc_lap + 3)):
            schedule[lap] = "SC"

    # Optionally trigger a stochastic SC or VSC event
    if sc_probability > 0.0 and total_laps >= 12 and rng.random() < sc_probability:
        event_type = "SC" if rng.random() < 0.65 else "VSC"
        duration = int(rng.integers(3, 5)) if event_type == "SC" else int(rng.integers(2, 4))
        start_lap = int(rng.integers(4, max(6, total_laps - 5)))

        for lap in range(start_lap, min(total_laps + 1, start_lap + duration)):
            if schedule[lap] == "NONE":
                schedule[lap] = event_type

    return schedule


def effective_pit_loss(base_pit_loss: float, track_status: str) -> float:
    """
    Calculate actual net time lost during a pit stop given track status.

    Under SC/VSC, field speed is restricted, cutting net pit loss significantly.
    """
    status = track_status.upper()
    if status == "SC":
        return round(base_pit_loss * 0.50, 2)  # ~50% cheaper pit stop
    if status == "VSC":
        return round(base_pit_loss * 0.65, 2)  # ~35% cheaper pit stop
    return round(base_pit_loss, 2)


def neutralized_lap_time(green_lap_time: float, track_status: str) -> float:
    """Adjust lap time when cars are slowed behind a Safety Car or VSC."""
    status = track_status.upper()
    if status == "SC":
        return round(green_lap_time * 1.38, 3)
    if status == "VSC":
        return round(green_lap_time * 1.22, 3)
    return green_lap_time


def evaluate_sc_pit_decision(
    current_lap: int,
    total_laps: int,
    tyre_age: int,
    next_planned_pit_lap: int | None,
    base_pit_loss: float = 22.0,
    track_status: str = "SC",
) -> dict[str, str | float]:
    """
    Recommend 'PIT NOW' or 'STAY OUT' when an SC or VSC is deployed.

    Logic:
    - Saves ~7-11s compared to a green-flag stop.
    - Worth boxing if tyre age >= 8 laps, or if within 8 laps of planned stop,
      provided at least 5 racing laps remain.
    """
    laps_remaining = total_laps - current_lap
    sc_pit_cost = effective_pit_loss(base_pit_loss, track_status)
    time_saved = round(base_pit_loss - sc_pit_cost, 2)

    if laps_remaining < 5:
        decision = "STAY OUT"
        reason = f"Only {laps_remaining} laps left; track position outweighs fresh tyres."
    elif next_planned_pit_lap is not None and abs(next_planned_pit_lap - current_lap) <= 8 and tyre_age >= 6:
        decision = "PIT NOW"
        reason = (
            f"Planned stop is close (Lap {next_planned_pit_lap}); boxing under {track_status} "
            f"saves ~{time_saved}s."
        )
    elif tyre_age >= 12 and laps_remaining >= 8:
        decision = "PIT NOW"
        reason = (
            f"Tyres are {tyre_age} laps old with {laps_remaining} laps left; "
            f"cheap {track_status} stop saves ~{time_saved}s."
        )
    else:
        decision = "STAY OUT"
        reason = f"Current tyres are only {tyre_age} laps old; preserve track position."

    return {
        "decision": decision,
        "time_saved_s": time_saved,
        "effective_pit_loss_s": sc_pit_cost,
        "reason": reason,
    }