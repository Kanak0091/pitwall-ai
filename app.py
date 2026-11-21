"""
Streamlit interactive race strategy dashboard for PitWall AI.

Run with:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_race_laps
from src.tyre_model import TyreDegradationModel
from src.weather import build_weather_timeline
from src.safety_car import generate_safety_car_schedule
from src.simulator import RaceSimulator, RaceStrategy
from src.optimizer import StrategyOptimizer
from src.monte_carlo import MonteCarloEngine

st.set_page_config(
    page_title="PitWall AI | F1 Race Strategy Simulator",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- F1 Pit-Wall Telemetry Color Palette ---
BG_PRIMARY = "#0B0E14"
BG_CARD = "#131720"
BORDER_SUBTLE = "#232A38"
F1_RED = "#E10600"
TEXT_MUTED = "#94A3B8"

COMPOUND_COLORS = {
    "SOFT": "#FF3B30",
    "MEDIUM": "#FFD60A",
    "HARD": "#E2E8F0",
    "INTERMEDIATE": "#32D74B",
    "WET": "#0A84FF",
}

PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def inject_custom_css() -> None:
    """Apply responsive F1 telemetry styling and auto-wrapping KPI cards."""
    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            padding-left: clamp(1rem, 3vw, 2.5rem);
            padding-right: clamp(1rem, 3vw, 2.5rem);
            max-width: 1440px;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {BG_CARD};
            border-right: 1px solid {BORDER_SUBTLE};
        }}
        .pitwall-banner {{
            background: linear-gradient(135deg, #161B26 0%, #10141D 100%);
            border: 1px solid {BORDER_SUBTLE};
            border-left: 5px solid {F1_RED};
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.5rem;
        }}
        .pitwall-title {{
            font-size: clamp(1.35rem, 2.5vw, 1.85rem);
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #F8FAFC;
            margin: 0;
            line-height: 1.25;
        }}
        .pitwall-subtitle {{
            color: {TEXT_MUTED};
            font-size: 0.92rem;
            margin-top: 0.4rem;
            margin-bottom: 0;
            line-height: 1.45;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }}
        .kpi-card {{
            background-color: {BG_CARD};
            border: 1px solid {BORDER_SUBTLE};
            border-radius: 12px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
            min-width: 0;
        }}
        .kpi-card.accent-red {{ border-top: 3px solid {F1_RED}; }}
        .kpi-card.accent-gold {{ border-top: 3px solid #FFD60A; }}
        .kpi-card.accent-cyan {{ border-top: 3px solid #38BDF8; }}
        .kpi-card.accent-green {{ border-top: 3px solid #32D74B; }}
        .kpi-label {{
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {TEXT_MUTED};
            font-weight: 600;
            margin-bottom: 0.45rem;
            white-space: nowrap;
        }}
        .kpi-value {{
            font-size: 1.25rem;
            font-weight: 700;
            color: #F8FAFC;
            line-height: 1.35;
        }}
        .kpi-sub {{
            font-size: 0.8rem;
            color: {TEXT_MUTED};
            margin-top: 0.35rem;
        }}
        hr {{
            border-color: {BORDER_SUBTLE};
            margin-top: 1.5rem;
            margin-bottom: 1.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_compound_badges(compounds: list[str]) -> str:
    """Render colored Pirelli tyre pills that wrap cleanly on smaller screens."""
    badges = []
    for comp in compounds:
        color = COMPOUND_COLORS.get(comp.upper(), "#E2E8F0")
        badges.append(
            f'<span style="display:inline-block; padding:2px 9px; margin:2px 0; '
            f'border-radius:999px; font-size:0.78rem; font-weight:700; white-space:nowrap; '
            f'color:#0B0E14; background-color:{color};">{comp}</span>'
        )
    return ' <span style="color:#94A3B8; font-weight:700;">→</span> '.join(badges)


@st.cache_data(show_spinner=False)
def get_cleaned_laps(year: int, grand_prix: str, use_offline_synthetic: bool) -> pd.DataFrame:
    """Load and cache race laps (FastF1 with synthetic fallback)."""
    if use_offline_synthetic:
        from src.data_loader import generate_synthetic_race_data
        return generate_synthetic_race_data()
    return load_race_laps(year=year, grand_prix=grand_prix, use_fallback=True)


@st.cache_data(show_spinner=False)
def run_cached_optimization(
    year: int,
    grand_prix: str,
    use_offline: bool,
    model_type: str,
    total_laps: int,
    base_pit_loss: float,
    starting_tyre: str,
    initial_weather: str,
    rain_prob: float,
    sc_prob: float,
    forced_sc_lap: int | None,
):
    """Cache tyre model fitting, simulation, and strategy optimization."""
    laps_df = get_cleaned_laps(year, grand_prix, use_offline)
    tyre_model = TyreDegradationModel(model_type=model_type).fit(laps_df)

    weather_timeline = build_weather_timeline(
        total_laps=total_laps,
        initial_condition=initial_weather,
        dynamic=rain_prob > 0.0,
        rain_probability=rain_prob,
        seed=42,
    )
    sc_schedule = generate_safety_car_schedule(
        total_laps=total_laps,
        sc_probability=sc_prob,
        forced_sc_lap=forced_sc_lap,
        seed=42,
    )

    simulator = RaceSimulator(
        tyre_model=tyre_model,
        total_laps=total_laps,
        base_pit_loss=base_pit_loss,
    )
    optimizer = StrategyOptimizer(simulator=simulator, min_stint_laps=6, step_size=2)

    start_filter = None if starting_tyre == "Any" else starting_tyre
    ranked_df, best_result = optimizer.optimize(
        starting_compound=start_filter,
        weather_timeline=weather_timeline,
        sc_schedule=sc_schedule,
    )
    best_by_stops = optimizer.best_per_stop_count(ranked_df)
    best_compounds = ranked_df.iloc[0]["Compounds"].split(" -> ")
    window_df, pit_windows = optimizer.compute_pit_window_heatmap(
        compounds=best_compounds[:3],
        weather_timeline=weather_timeline,
        sc_schedule=sc_schedule,
    )
    return tyre_model, simulator, ranked_df, best_result, best_by_stops, window_df, pit_windows


def format_race_time(seconds: float) -> str:
    """Format total seconds into H:MM:SS.sss race clock string."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs}:{mins:02d}:{secs:06.3f}"


def style_plotly_figure(fig: go.Figure, title: str, x_title: str, y_title: str) -> go.Figure:
    """Apply consistent PitWall dark telemetry styling and clean margins."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#F8FAFC"), x=0.02, y=0.95),
        xaxis_title=x_title,
        yaxis_title=y_title,
        paper_bgcolor=BG_CARD,
        plot_bgcolor="#0E121B",
        font=dict(color="#CBD5E1", size=12),
        height=350,
        margin=dict(l=50, r=20, t=60, b=45),
        xaxis=dict(gridcolor="#1E2533", zerolinecolor="#1E2533"),
        yaxis=dict(gridcolor="#1E2533", zerolinecolor="#1E2533"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    return fig


def plot_tyre_degradation(tyre_model: TyreDegradationModel, max_age: int = 35) -> go.Figure:
    """Plot fuel-corrected degradation curves for Soft, Medium, and Hard tyres."""
    ages = np.arange(1, max_age + 1)
    fig = go.Figure()

    for compound in ["SOFT", "MEDIUM", "HARD"]:
        cm = tyre_model.get_model(compound)
        paces = [
            cm.predict_lap_time(tyre_age=int(age), lap_number=57, total_laps=57)
            for age in ages
        ]
        fig.add_trace(
            go.Scatter(
                x=ages,
                y=paces,
                mode="lines",
                name=f"{compound} (+{cm.deg_per_lap:.3f}s/lap)",
                line=dict(color=COMPOUND_COLORS[compound], width=3),
            )
        )

    return style_plotly_figure(
        fig,
        title="Fuel-Corrected Tyre Degradation Curves",
        x_title="Tyre Age (Laps)",
        y_title="Zero-Fuel Lap Time (s)",
    )


def plot_lap_time_trace(lap_df: pd.DataFrame, strategy_name: str) -> go.Figure:
    """Plot lap-by-lap predicted race pace with compound colours and pit markers."""
    fig = go.Figure()

    for compound, group in lap_df.groupby("Compound"):
        fig.add_trace(
            go.Scatter(
                x=group["Lap"],
                y=group["LapTime_s"],
                mode="markers+lines",
                name=compound,
                marker=dict(color=COMPOUND_COLORS.get(compound, "#FFFFFF"), size=5),
                line=dict(color=COMPOUND_COLORS.get(compound, "#FFFFFF"), width=1.8),
            )
        )

    pit_laps = lap_df[lap_df["PitStop"]]["Lap"].tolist()
    for p_lap in pit_laps:
        fig.add_vline(
            x=p_lap,
            line_dash="dash",
            line_color=F1_RED,
            annotation_text=f"BOX L{p_lap}",
            annotation_font_color="#FCA5A5",
            annotation_position="top right",
        )

    return style_plotly_figure(
        fig,
        title=f"Lap-by-Lap Pace Trace — {strategy_name}",
        x_title="Lap Number",
        y_title="Predicted Lap Time (s)",
    )


def plot_strategy_comparison(best_by_stops: pd.DataFrame) -> go.Figure:
    """Bar chart comparing the fastest 1-stop, 2-stop, and 3-stop strategies."""
    df = best_by_stops.copy()
    df["StopLabel"] = df["Stops"].astype(str) + "-Stop (" + df["Compounds"] + ")"
    bar_colors = ["#38BDF8", "#FFD60A", "#FF3B30"][: len(df)]

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["StopLabel"],
                y=df["DeltaToBest_s"],
                text=[f"+{v:.2f}s" for v in df["DeltaToBest_s"]],
                textposition="outside",
                marker_color=bar_colors,
                width=0.45,
            )
        ]
    )
    max_y = max(df["DeltaToBest_s"].max() * 1.25, 1.5)
    fig.update_yaxes(range=[0, max_y])

    return style_plotly_figure(
        fig,
        title="1-Stop vs 2-Stop vs 3-Stop (Delta to Optimal)",
        x_title="Strategy Configuration",
        y_title="Time Behind Optimal (s)",
    )


def plot_pit_window_heatmap(window_df: pd.DataFrame, compounds: list[str]) -> go.Figure:
    """Lightweight go.Heatmap/go.Bar pit window visualization."""
    if len(compounds) == 2:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=window_df["PitLap1"],
                    y=window_df["Delta_s"],
                    marker_color="#38BDF8",
                )
            ]
        )
        return style_plotly_figure(
            fig,
            title=f"Pit Window Sensitivity ({' → '.join(compounds)})",
            x_title="Pit Stop Lap",
            y_title="Time Loss vs Optimal Lap (s)",
        )

    pivot = window_df.pivot(index="PitLap1", columns="PitLap2", values="Delta_s")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale=[[0.0, "#10B981"], [0.45, "#F59E0B"], [1.0, "#EF4444"]],
            colorbar=dict(title="Delta (s)"),
        )
    )
    return style_plotly_figure(
        fig,
        title=f"2-Stop Pit Window Heatmap ({' → '.join(compounds)})",
        x_title="Second Pit Stop Lap",
        y_title="First Pit Stop Lap",
    )


def main() -> None:
    inject_custom_css()

    # --- Top Telemetry Header Banner ---
    st.markdown(
        """
        <div class="pitwall-banner">
            <h1 class="pitwall-title">🏎️ PitWall AI — Formula 1 Race Strategy Simulator</h1>
            <p class="pitwall-subtitle">
                Lap-by-lap race strategy engine modeling fuel-corrected tyre degradation,
                dynamic weather transitions, Safety Car pit windows, and Monte Carlo win probabilities.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar Controls ---
    st.sidebar.markdown("### ⚙️ Race Configuration")
    data_source = st.sidebar.radio(
        "Telemetry Source",
        options=["Synthetic Offline Data (Instant)", "FastF1 Real Grand Prix Data"],
        index=0,
    )
    use_offline = data_source.startswith("Synthetic")

    year = st.sidebar.selectbox("Season Year", [2024, 2023, 2022], index=0, disabled=use_offline)
    grand_prix = st.sidebar.selectbox(
        "Grand Prix",
        ["Bahrain", "Monza", "Silverstone", "Spa", "Suzuka", "Austin"],
        index=0,
        disabled=use_offline,
    )

    total_laps = st.sidebar.slider("Total Race Laps", min_value=30, max_value=72, value=57)
    starting_tyre = st.sidebar.selectbox("Starting Tyre Compound", ["Any", "SOFT", "MEDIUM", "HARD"], index=2)
    model_type = st.sidebar.selectbox(
        "Tyre Degradation Model",
        options=["linear", "gbr"],
        format_func=lambda x: "Linear Regression" if x == "linear" else "Gradient Boosting (Non-Linear)",
    )
    base_pit_loss = st.sidebar.slider("Green-Flag Pit Stop Loss (s)", 18.0, 28.0, 22.0, 0.5)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌦️ Track & Chaos Events")
    initial_weather = st.sidebar.selectbox("Initial Weather", ["dry", "damp", "wet"], index=0)
    rain_prob = st.sidebar.slider("Mid-Race Rain Shower Probability", 0.0, 1.0, 0.0, 0.05)
    sc_prob = st.sidebar.slider("Safety Car / VSC Probability", 0.0, 1.0, 0.35, 0.05)
    trigger_sc = st.sidebar.checkbox("Force Safety Car Deployment", value=False)
    forced_sc_lap = (
        st.sidebar.slider("Forced Safety Car Lap", 5, total_laps - 5, total_laps // 3)
        if trigger_sc
        else None
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎲 Monte Carlo Settings")
    mc_runs = st.sidebar.select_slider("Monte Carlo Simulations", options=[200, 500, 1000, 2000], value=1000)

    # --- Cached Optimization & Simulation ---
    (
        tyre_model,
        simulator,
        ranked_df,
        best_result,
        best_by_stops,
        window_df,
        pit_windows,
    ) = run_cached_optimization(
        year=year,
        grand_prix=grand_prix,
        use_offline=use_offline,
        model_type=model_type,
        total_laps=total_laps,
        base_pit_loss=base_pit_loss,
        starting_tyre=starting_tyre,
        initial_weather=initial_weather,
        rain_prob=rain_prob,
        sc_prob=sc_prob,
        forced_sc_lap=forced_sc_lap,
    )

    # --- Recommended Strategy Responsive KPI Grid ---
    st.markdown("### 🏆 Recommended Race Strategy")
    runner_up_delta = float(ranked_df.iloc[1]["DeltaToBest_s"]) if len(ranked_df) > 1 else 0.0
    best_compounds = ranked_df.iloc[0]["Compounds"].split(" -> ")
    best_pit_laps = ranked_df.iloc[0]["PlannedPitLaps"]

    window_str = " & ".join([f"Laps {w[0]}–{w[1]}" for w in pit_windows])
    exact_pit_str = ", ".join([f"Lap {p}" for p in best_pit_laps])
    badges_html = render_compound_badges(best_compounds)
    runner_up_name = ranked_df.iloc[1]["Compounds"] if len(ranked_df) > 1 else "N/A"

    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card accent-red">
                <div class="kpi-label">Optimal Tyre Sequence</div>
                <div class="kpi-value" style="font-size:1rem;">{badges_html}</div>
                <div class="kpi-sub">{len(best_pit_laps)}-Stop Strategy</div>
            </div>
            <div class="kpi-card accent-gold">
                <div class="kpi-label">Optimal Pit Window</div>
                <div class="kpi-value">{window_str}</div>
                <div class="kpi-sub">Target Box: {exact_pit_str}</div>
            </div>
            <div class="kpi-card accent-cyan">
                <div class="kpi-label">Predicted Race Time</div>
                <div class="kpi-value">{format_race_time(best_result.total_time_s)}</div>
                <div class="kpi-sub">{total_laps} Laps Total Distance</div>
            </div>
            <div class="kpi-card accent-green">
                <div class="kpi-label">Advantage Over #2</div>
                <div class="kpi-value">+{runner_up_delta:.2f} s</div>
                <div class="kpi-sub">vs {runner_up_name}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if best_result.sc_recommendations:
        for rec in best_result.sc_recommendations:
            st.info(
                f"🚨 **{rec['track_status']} Deployed on Lap {rec['lap']}** → "
                f"**{rec['decision']}** — {rec['reason']}"
            )

    st.divider()

    # --- Tabbed Charts (cuts initial JS paint cost in half) ---
    tab1, tab2 = st.tabs(["📈 Pace & Tyre Degradation", "🔥 Strategy Comparison & Pit Windows"])

    with tab1:
        col_left, col_right = st.columns(2, gap="large")
        with col_left:
            st.plotly_chart(
                plot_tyre_degradation(tyre_model, max_age=total_laps // 2 + 5),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        with col_right:
            st.plotly_chart(
                plot_lap_time_trace(best_result.lap_history, best_result.strategy_name),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )

    with tab2:
        col_left2, col_right2 = st.columns(2, gap="large")
        with col_left2:
            st.plotly_chart(
                plot_strategy_comparison(best_by_stops),
                use_container_width=True,
                config=PLOTLY_CONFIG,
            )
        with col_right2:
            if len(best_compounds) in {2, 3}:
                st.plotly_chart(
                    plot_pit_window_heatmap(window_df, best_compounds),
                    use_container_width=True,
                    config=PLOTLY_CONFIG,
                )
            else:
                st.info("Pit-window heatmap is displayed for 1-stop and 2-stop strategies.")

    st.divider()

    # --- Top Strategies Leaderboard ---
    st.markdown("### 📋 Top 10 Ranked Strategies")
    st.dataframe(
        ranked_df.head(10)[["Rank", "Strategy", "Stops", "Compounds", "TotalTime_s", "DeltaToBest_s"]],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --- Monte Carlo Simulation Section ---
    st.markdown(f"### 🎲 Monte Carlo Robustness Analysis ({mc_runs:,} Simulated Races)")
    st.caption(
        "Evaluates top candidate strategies across randomized Safety Cars, "
        "weather showers, and lap-time variance."
    )

    if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
        top_candidates = []
        for _, row in best_by_stops.iterrows():
            top_candidates.append(
                RaceStrategy(
                    name=row["Strategy"],
                    compounds=row["Compounds"].split(" -> "),
                    pit_laps=row["PlannedPitLaps"],
                )
            )
        if len(ranked_df) > 1:
            second_row = ranked_df.iloc[1]
            if second_row["Strategy"] not in {s.name for s in top_candidates}:
                top_candidates.append(
                    RaceStrategy(
                        name=second_row["Strategy"],
                        compounds=second_row["Compounds"].split(" -> "),
                        pit_laps=second_row["PlannedPitLaps"],
                    )
                )

        with st.spinner(f"Simulating {mc_runs:,} races..."):
            mc_engine = MonteCarloEngine(simulator=simulator, n_simulations=mc_runs)
            mc_out = mc_engine.run(
                strategies=top_candidates,
                initial_weather=initial_weather,
                rain_probability=max(rain_prob, 0.15),
                sc_probability=max(sc_prob, 0.35),
            )

        mc_col1, mc_col2 = st.columns([1.25, 1], gap="large")
        with mc_col1:
            st.dataframe(mc_out.summary_df, use_container_width=True, hide_index=True)
        with mc_col2:
            fig_mc = go.Figure(
                data=[
                    go.Bar(
                        x=mc_out.summary_df["WinProb_%"],
                        y=mc_out.summary_df["Strategy"],
                        orientation="h",
                        text=[f"{v:.1f}%" for v in mc_out.summary_df["WinProb_%"]],
                        textposition="outside",
                        marker_color="#E10600",
                    )
                ]
            )
            fig_mc = style_plotly_figure(
                fig_mc,
                title="Win Probability by Strategy (%)",
                x_title="Win Probability (%)",
                y_title="",
            )
            fig_mc.update_layout(height=320)
            st.plotly_chart(fig_mc, use_container_width=True, config=PLOTLY_CONFIG)


if __name__ == "__main__":
    main()
