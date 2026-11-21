"""
Data ingestion and cleaning module for PitWall AI.

Loads real F1 race lap data via FastF1 and cleans out pit laps,
non-green-flag laps, and outliers. Includes a realistic synthetic
data generator as an offline fallback.
"""

from pathlib import Path
import logging
import numpy as np
import pandas as pd

try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False

logger = logging.getLogger(__name__)

VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}


def enable_fastf1_cache(cache_dir: str = "data/fastf1_cache") -> None:
    """Create and enable the local FastF1 cache directory."""
    if not FASTF1_AVAILABLE:
        return
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))


def load_race_laps(
    year: int = 2024,
    grand_prix: str = "Bahrain",
    driver: str | None = None,
    use_fallback: bool = True,
) -> pd.DataFrame:
    """
    Load and clean green-flag race laps for a given Grand Prix.

    Args:
        year: Championship season year (e.g., 2024).
        grand_prix: Race name or location (e.g., "Bahrain", "Monza").
        driver: Optional 3-letter driver code (e.g., "VER", "LEC").
                If None, loads all drivers for richer tyre-wear training.
        use_fallback: Return synthetic data if FastF1 fails or is offline.

    Returns:
        Cleaned DataFrame with columns:
        ['Driver', 'LapNumber', 'LapTimeSeconds', 'Compound', 'TyreLife', 'TotalLaps']
    """
    if not FASTF1_AVAILABLE:
        if use_fallback:
            logger.warning("FastF1 not installed. Using synthetic fallback data.")
            return generate_synthetic_race_data()
        raise ImportError("fastf1 is not installed.")

    try:
        enable_fastf1_cache()
        session = fastf1.get_session(year, grand_prix, "R")
        session.load(telemetry=False, weather=False, messages=False)
        laps = session.laps.copy()

        if driver:
            laps = laps.pick_drivers(driver)

        return clean_laps(laps)

    except Exception as exc:
        if use_fallback:
            logger.warning(f"FastF1 load failed ({exc}). Falling back to synthetic data.")
            return generate_synthetic_race_data()
        raise


def clean_laps(raw_laps: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out pit laps, safety car laps, missing values, and slow outliers.

    Args:
        raw_laps: Raw laps DataFrame from FastF1.

    Returns:
        Cleaned DataFrame ready for tyre degradation modeling.
    """
    df = raw_laps.copy()
    total_laps = int(df["LapNumber"].max()) if not df.empty else 57

    # 1. Remove pit in-laps and out-laps
    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]

    # 2. Keep only green-flag laps (TrackStatus == '1')
    if "TrackStatus" in df.columns:
        df = df[df["TrackStatus"].astype(str) == "1"]

    # 3. Drop missing lap times or unknown tyre compounds
    df = df.dropna(subset=["LapTime", "Compound", "TyreLife", "LapNumber"])
    df["Compound"] = df["Compound"].str.upper()
    df = df[df["Compound"].isin(VALID_COMPOUNDS)]

    # 4. Convert timedelta LapTime to float seconds
    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

    # 5. Remove outliers (> 107% of median race pace)
    median_pace = df["LapTimeSeconds"].median()
    df = df[df["LapTimeSeconds"] <= median_pace * 1.07]

    df["TotalLaps"] = total_laps

    columns = ["Driver", "LapNumber", "LapTimeSeconds", "Compound", "TyreLife", "TotalLaps"]
    return df[columns].reset_index(drop=True)


def generate_synthetic_race_data(
    total_laps: int = 57,
    base_pace: float = 92.5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic F1 lap data across multiple drivers and compounds.

    Includes true fuel burn effect (~0.032s faster per lap as fuel burns off)
    and compound-specific tyre degradation plus realistic Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    records = []

    compound_specs = {
        "SOFT": {"offset": -0.9, "deg_per_lap": 0.095},
        "MEDIUM": {"offset": 0.0, "deg_per_lap": 0.055},
        "HARD": {"offset": 0.6, "deg_per_lap": 0.030},
    }

    stint_plans = [
        ("DRV1", [("SOFT", 16), ("HARD", 22), ("MEDIUM", 19)]),
        ("DRV2", [("MEDIUM", 22), ("HARD", 35)]),
        ("DRV3", [("SOFT", 14), ("MEDIUM", 21), ("SOFT", 22)]),
        ("DRV4", [("MEDIUM", 25), ("HARD", 32)]),
        ("DRV5", [("HARD", 30), ("MEDIUM", 27)]),
        ("DRV6", [("SOFT", 15), ("HARD", 25), ("HARD", 17)]),
    ]

    fuel_effect_per_lap = 0.032

    for driver_code, stints in stint_plans:
        lap_num = 1
        driver_skill_delta = rng.normal(0.0, 0.25)

        for compound, stint_len in stints:
            spec = compound_specs[compound]
            for tyre_age in range(1, stint_len + 1):
                if lap_num > total_laps:
                    break

                if tyre_age == 1 and lap_num > 1:
                    lap_num += 1
                    continue

                laps_remaining = total_laps - lap_num
                fuel_penalty = laps_remaining * fuel_effect_per_lap
                tyre_wear = tyre_age * spec["deg_per_lap"]
                noise = rng.normal(0.0, 0.18)

                lap_time = (
                    base_pace
                    + driver_skill_delta
                    + spec["offset"]
                    + fuel_penalty
                    + tyre_wear
                    + noise
                )

                records.append(
                    {
                        "Driver": driver_code,
                        "LapNumber": lap_num,
                        "LapTimeSeconds": round(lap_time, 3),
                        "Compound": compound,
                        "TyreLife": tyre_age,
                        "TotalLaps": total_laps,
                    }
                )
                lap_num += 1

    return pd.DataFrame(records)