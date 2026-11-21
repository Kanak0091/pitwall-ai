"""
Tyre degradation and fuel-corrected pace modeling for PitWall AI.

Isolates true tyre wear by removing fuel-burn effects from raw lap times,
then fits compound-specific regression models (Linear or Gradient Boosting).
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

DEFAULT_FUEL_EFFECT_PER_LAP = 0.032

DEFAULT_COMPOUND_PARAMS = {
    "SOFT": {"base_offset": -0.85, "deg_per_lap": 0.090},
    "MEDIUM": {"base_offset": 0.00, "deg_per_lap": 0.055},
    "HARD": {"base_offset": 0.55, "deg_per_lap": 0.030},
    "INTERMEDIATE": {"base_offset": 4.50, "deg_per_lap": 0.065},
    "WET": {"base_offset": 9.00, "deg_per_lap": 0.050},
}


@dataclass
class CompoundModel:
    """Stores fitted degradation parameters and ML predictor for one tyre compound."""
    compound: str
    base_pace: float
    deg_per_lap: float
    model_type: str
    estimator: object | None = None

    def predict_lap_time(
        self,
        tyre_age: int,
        lap_number: int,
        total_laps: int,
        fuel_effect_per_lap: float = DEFAULT_FUEL_EFFECT_PER_LAP,
    ) -> float:
        """Predict lap time given tyre age and current race fuel load."""
        laps_remaining = max(0, total_laps - lap_number)
        fuel_penalty = laps_remaining * fuel_effect_per_lap

        if self.estimator is not None and self.model_type == "gbr":
            zero_fuel_pace = float(self.estimator.predict(np.array([[tyre_age]]))[0])
        else:
            zero_fuel_pace = self.base_pace + (self.deg_per_lap * max(0, tyre_age - 1))

        return round(zero_fuel_pace + fuel_penalty, 3)


class TyreDegradationModel:
    """Trains and queries tyre degradation models across all F1 compounds."""

    def __init__(
        self,
        model_type: str = "linear",
        fuel_effect_per_lap: float = DEFAULT_FUEL_EFFECT_PER_LAP,
    ) -> None:
        if model_type not in {"linear", "gbr"}:
            raise ValueError("model_type must be 'linear' or 'gbr'")
        self.model_type = model_type
        self.fuel_effect_per_lap = fuel_effect_per_lap
        self.models: dict[str, CompoundModel] = {}
        self.reference_medium_pace: float = 92.5

    def apply_fuel_correction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'FuelCorrectedLapTime' column representing zero-fuel pace.

        Formula:
            FuelCorrectedLapTime = LapTimeSeconds - (TotalLaps - LapNumber) * fuel_effect
        """
        corrected = df.copy()
        laps_remaining = (corrected["TotalLaps"] - corrected["LapNumber"]).clip(lower=0)
        corrected["FuelCorrectedLapTime"] = (
            corrected["LapTimeSeconds"] - laps_remaining * self.fuel_effect_per_lap
        )
        return corrected

    def fit(self, laps_df: pd.DataFrame) -> "TyreDegradationModel":
        """Fit a degradation model for each compound present in the cleaned laps."""
        df = self.apply_fuel_correction(laps_df)

        med_df = df[df["Compound"] == "MEDIUM"]
        if not med_df.empty:
            self.reference_medium_pace = float(med_df["FuelCorrectedLapTime"].quantile(0.25))
        else:
            self.reference_medium_pace = float(df["FuelCorrectedLapTime"].quantile(0.25))

        for compound, defaults in DEFAULT_COMPOUND_PARAMS.items():
            sub = df[df["Compound"] == compound]

            if len(sub) >= 6 and sub["TyreLife"].nunique() >= 3:
                X = sub[["TyreLife"]].values
                y = sub["FuelCorrectedLapTime"].values

                lin = LinearRegression().fit(X, y)
                slope = max(0.015, float(lin.coef_[0]))
                fresh_pace = float(lin.intercept_ + slope)

                estimator = None
                if self.model_type == "gbr":
                    estimator = GradientBoostingRegressor(
                        n_estimators=60, max_depth=2, learning_rate=0.08, random_state=42
                    )
                    estimator.fit(X, y)

                self.models[compound] = CompoundModel(
                    compound=compound,
                    base_pace=round(fresh_pace, 3),
                    deg_per_lap=round(slope, 4),
                    model_type=self.model_type,
                    estimator=estimator,
                )
            else:
                self.models[compound] = CompoundModel(
                    compound=compound,
                    base_pace=round(self.reference_medium_pace + defaults["base_offset"], 3),
                    deg_per_lap=defaults["deg_per_lap"],
                    model_type="linear",
                    estimator=None,
                )

        return self

    def get_model(self, compound: str) -> CompoundModel:
        """Retrieve the fitted CompoundModel for a given tyre compound."""
        comp = compound.upper()
        if comp not in self.models:
            raise KeyError(f"Unknown tyre compound: {compound}")
        return self.models[comp]

    def summary_table(self) -> pd.DataFrame:
        """Return a readable summary DataFrame of fitted tyre wear rates."""
        rows = [
            {
                "Compound": m.compound,
                "FreshPace_s": m.base_pace,
                "DegPerLap_s": m.deg_per_lap,
                "ModelType": m.model_type,
            }
            for m in self.models.values()
        ]
        return pd.DataFrame(rows)