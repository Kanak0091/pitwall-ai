"""
High-performance FastAPI + Zero-Dependency HTML5 Canvas Telemetry UI for PitWall AI.
Achieves 90+ Google Lighthouse Performance while reusing the exact src/ simulation engine.

Run with:
    uvicorn web_app:app --host 0.0.0.0 --port 8000
"""

from functools import lru_cache
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from src.data_loader import generate_synthetic_race_data, load_race_laps
from src.tyre_model import TyreDegradationModel
from src.weather import build_weather_timeline
from src.safety_car import generate_safety_car_schedule
from src.simulator import RaceSimulator, RaceStrategy
from src.optimizer import StrategyOptimizer
from src.monte_carlo import MonteCarloEngine

app = FastAPI(title="PitWall AI — High-Performance Telemetry Engine")


@lru_cache(maxsize=16)
def get_fitted_model(use_offline: bool, year: int, grand_prix: str, model_type: str) -> TyreDegradationModel:
    laps = (
        generate_synthetic_race_data()
        if use_offline
        else load_race_laps(year=year, grand_prix=grand_prix, use_fallback=True)
    )
    return TyreDegradationModel(model_type=model_type).fit(laps)


def format_race_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs}:{mins:02d}:{secs:06.3f}"


@app.get("/api/simulate")
def simulate_endpoint(
    total_laps: int = Query(57, ge=30, le=72),
    starting_tyre: str = Query("MEDIUM"),
    model_type: str = Query("linear"),
    base_pit_loss: float = Query(22.0, ge=18.0, le=28.0),
    initial_weather: str = Query("dry"),
    rain_prob: float = Query(0.0, ge=0.0, le=1.0),
    sc_prob: float = Query(0.35, ge=0.0, le=1.0),
    forced_sc_lap: int = Query(0, ge=0, le=70),
    run_mc: bool = Query(False),
    mc_runs: int = Query(500, ge=100, le=2000),
):
    tyre_model = get_fitted_model(True, 2024, "Bahrain", model_type)
    sc_lap = forced_sc_lap if forced_sc_lap > 0 else None

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
        forced_sc_lap=sc_lap,
        seed=42,
    )

    sim = RaceSimulator(tyre_model=tyre_model, total_laps=total_laps, base_pit_loss=base_pit_loss)
    opt = StrategyOptimizer(simulator=sim, min_stint_laps=6, step_size=2)

    start_filter = None if starting_tyre == "Any" else starting_tyre
    ranked_df, best_res = opt.optimize(
        starting_compound=start_filter,
        weather_timeline=weather_timeline,
        sc_schedule=sc_schedule,
    )
    best_by_stops = opt.best_per_stop_count(ranked_df)
    best_compounds = ranked_df.iloc[0]["Compounds"].split(" -> ")
    _, pit_windows = opt.compute_pit_window_heatmap(
        compounds=best_compounds[:3],
        weather_timeline=weather_timeline,
        sc_schedule=sc_schedule,
    )

    # Degradation curves for Canvas chart
    max_age = total_laps // 2 + 5
    deg_curves = {}
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        cm = tyre_model.get_model(comp)
        deg_curves[comp] = {
            "slope": round(cm.deg_per_lap, 4),
            "points": [
                round(cm.predict_lap_time(tyre_age=age, lap_number=total_laps, total_laps=total_laps), 3)
                for age in range(1, max_age + 1)
            ],
        }

    mc_summary = []
    if run_mc:
        candidates = [
            RaceStrategy(
                name=row["Strategy"],
                compounds=row["Compounds"].split(" -> "),
                pit_laps=row["PlannedPitLaps"],
            )
            for _, row in best_by_stops.iterrows()
        ]
        mc = MonteCarloEngine(simulator=sim, n_simulations=mc_runs)
        mc_out = mc.run(
            strategies=candidates,
            initial_weather=initial_weather,
            rain_probability=max(rain_prob, 0.15),
            sc_probability=max(sc_prob, 0.35),
        )
        mc_summary = mc_out.summary_df.to_dict(orient="records")

    runner_up_delta = float(ranked_df.iloc[1]["DeltaToBest_s"]) if len(ranked_df) > 1 else 0.0
    runner_up_comp = ranked_df.iloc[1]["Compounds"] if len(ranked_df) > 1 else "N/A"

    return JSONResponse(
        {
            "best_compounds": best_compounds,
            "best_pit_laps": ranked_df.iloc[0]["PlannedPitLaps"],
            "pit_windows": [f"Laps {w[0]}–{w[1]}" for w in pit_windows],
            "predicted_time": format_race_time(best_res.total_time_s),
            "total_laps": total_laps,
            "advantage_s": round(runner_up_delta, 2),
            "runner_up": runner_up_comp,
            "sc_advice": best_res.sc_recommendations,
            "deg_curves": deg_curves,
            "lap_trace": best_res.lap_history[["Lap", "LapTime_s", "Compound", "PitStop"]].to_dict(orient="records"),
            "stop_comparison": best_by_stops[["Stops", "Compounds", "TotalTime_s", "DeltaToBest_s"]].to_dict(orient="records"),
            "leaderboard": ranked_df.head(8)[["Rank", "Strategy", "Stops", "Compounds", "TotalTime_s", "DeltaToBest_s"]].to_dict(orient="records"),
            "mc_summary": mc_summary,
        }
    )


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="PitWall AI — Formula 1 Race Strategy & Monte Carlo Simulator" />
  <title>PitWall AI | F1 Race Strategy Simulator</title>
  <style>
    :root {
      --bg: #0B0E14;
      --card: #131720;
      --border: #232A38;
      --red: #E10600;
      --text: #F8FAFC;
      --muted: #94A3B8;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    .layout {
      display: grid;
      grid-template-columns: 300px 1fr;
      min-height: 100vh;
    }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
    }
    aside {
      background: var(--card);
      border-right: 1px solid var(--border);
      padding: 1.25rem;
    }
    main {
      padding: 1.5rem clamp(1rem, 3vw, 2.25rem);
      max-width: 1380px;
      width: 100%;
    }
    h1 { font-size: clamp(1.3rem, 2.3vw, 1.8rem); margin: 0; font-weight: 800; }
    h2 { font-size: 1.15rem; margin: 0 0 0.85rem; }
    .banner {
      background: linear-gradient(135deg, #161B26 0%, #10141D 100%);
      border: 1px solid var(--border);
      border-left: 5px solid var(--red);
      border-radius: 12px;
      padding: 1.2rem 1.4rem;
      margin-bottom: 1.5rem;
    }
    .banner p { color: var(--muted); margin: 0.35rem 0 0; font-size: 0.92rem; }
    .control { margin-bottom: 1rem; }
    .control label {
      display: flex; justify-content: space-between;
      font-size: 0.82rem; color: #CBD5E1; margin-bottom: 0.35rem; font-weight: 600;
    }
    select, input[type="range"] { width: 100%; accent-color: var(--red); }
    select {
      background: #0B0E14; color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: 0.48rem 0.6rem; font-size: 0.88rem;
    }
    .btn {
      width: 100%; background: var(--red); color: #fff; border: none;
      border-radius: 8px; padding: 0.65rem 1rem; font-weight: 700; cursor: pointer;
      font-size: 0.9rem; margin-top: 0.4rem;
    }
    .btn:hover { filter: brightness(1.1); }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 1rem;
      margin-bottom: 1.5rem;
    }
    .kpi {
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 1.05rem 1.2rem;
    }
    .kpi.red { border-top: 3px solid var(--red); }
    .kpi.gold { border-top: 3px solid #FFD60A; }
    .kpi.cyan { border-top: 3px solid #38BDF8; }
    .kpi.green { border-top: 3px solid #32D74B; }
    .kpi-lbl { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 700; }
    .kpi-val { font-size: 1.25rem; font-weight: 800; margin: 0.35rem 0 0.25rem; }
    .kpi-sub { font-size: 0.8rem; color: var(--muted); }
    .pill {
      display: inline-block; padding: 2px 9px; border-radius: 999px;
      font-size: 0.78rem; font-weight: 800; color: #0B0E14;
    }
    .charts-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1.25rem; margin-bottom: 1.5rem;
    }
    .chart-card {
      background: var(--card); border: 1px solid var(--border);
      border-radius: 12px; padding: 1rem;
    }
    .chart-title { font-size: 0.92rem; font-weight: 700; margin-bottom: 0.6rem; }
    canvas { width: 100%; height: 260px; display: block; background: #0E121B; border-radius: 8px; }
    table {
      width: 100%; border-collapse: collapse; background: var(--card);
      border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
      font-size: 0.86rem; margin-bottom: 1.5rem;
    }
    th, td { padding: 0.65rem 0.85rem; text-align: left; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); font-size: 0.76rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .alert {
      background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.35);
      border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 1.25rem; font-size: 0.88rem;
    }
  </style>
</head>
<body>
<div class="layout">
  <aside>
    <h2>⚙️ Race Configuration</h2>
    <div class="control">
      <label>Total Race Laps <span id="lapsVal">57</span></label>
      <input type="range" id="totalLaps" min="30" max="72" value="57" oninput="lapsVal.textContent=this.value" onchange="updateSim(false)">
    </div>
    <div class="control">
      <label>Starting Tyre Compound</label>
      <select id="startTyre" onchange="updateSim(false)">
        <option value="Any">Any</option>
        <option value="SOFT">SOFT</option>
        <option value="MEDIUM" selected>MEDIUM</option>
        <option value="HARD">HARD</option>
      </select>
    </div>
    <div class="control">
      <label>Degradation Model</label>
      <select id="modelType" onchange="updateSim(false)">
        <option value="linear" selected>Linear Regression</option>
        <option value="gbr">Gradient Boosting</option>
      </select>
    </div>
    <div class="control">
      <label>Pit Stop Loss (s) <span id="pitVal">22.0</span></label>
      <input type="range" id="pitLoss" min="18" max="28" step="0.5" value="22" oninput="pitVal.textContent=Number(this.value).toFixed(1)" onchange="updateSim(false)">
    </div>
    <div class="control">
      <label>Initial Weather</label>
      <select id="weather" onchange="updateSim(false)">
        <option value="dry" selected>Dry</option>
        <option value="damp">Damp</option>
        <option value="wet">Wet</option>
      </select>
    </div>
    <div class="control">
      <label>Rain Probability <span id="rainVal">0%</span></label>
      <input type="range" id="rainProb" min="0" max="1" step="0.05" value="0" oninput="rainVal.textContent=Math.round(this.value*100)+'%'" onchange="updateSim(false)">
    </div>
    <div class="control">
      <label>Safety Car Probability <span id="scVal">35%</span></label>
      <input type="range" id="scProb" min="0" max="1" step="0.05" value="0.35" oninput="scVal.textContent=Math.round(this.value*100)+'%'" onchange="updateSim(false)">
    </div>
    <button class="btn" onclick="updateSim(true)">🎲 Run 500 Monte Carlo Races</button>
  </aside>

  <main>
    <div class="banner">
      <h1>🏎️ PitWall AI — Formula 1 Race Strategy Simulator</h1>
      <p>Fuel-corrected tyre degradation, dynamic weather transitions, Safety Car pit windows, and Monte Carlo win probabilities.</p>
    </div>

    <h2>🏆 Recommended Race Strategy</h2>
    <div class="kpi-grid">
      <div class="kpi red">
        <div class="kpi-lbl">Optimal Tyre Sequence</div>
        <div class="kpi-val" id="kpiSeq">Loading...</div>
        <div class="kpi-sub" id="kpiStops">—</div>
      </div>
      <div class="kpi gold">
        <div class="kpi-lbl">Optimal Pit Window</div>
        <div class="kpi-val" id="kpiWin">—</div>
        <div class="kpi-sub" id="kpiExact">—</div>
      </div>
      <div class="kpi cyan">
        <div class="kpi-lbl">Predicted Race Time</div>
        <div class="kpi-val" id="kpiTime">—</div>
        <div class="kpi-sub" id="kpiDist">—</div>
      </div>
      <div class="kpi green">
        <div class="kpi-lbl">Advantage Over #2</div>
        <div class="kpi-val" id="kpiAdv">—</div>
        <div class="kpi-sub" id="kpiRunner">—</div>
      </div>
    </div>

    <div id="scAlert"></div>

    <div class="charts-grid">
      <div class="chart-card">
        <div class="chart-title">Fuel-Corrected Tyre Degradation Curves (Zero-Fuel s)</div>
        <canvas id="degCanvas" width="560" height="260"></canvas>
      </div>
      <div class="chart-card">
        <div class="chart-title">Lap-by-Lap Race Pace Trace (s)</div>
        <canvas id="paceCanvas" width="560" height="260"></canvas>
      </div>
    </div>

    <h2>📋 Top Ranked Race Strategies</h2>
    <table>
      <thead>
        <tr><th>Rank</th><th>Strategy</th><th>Stops</th><th>Compounds</th><th>Total Time (s)</th><th>Delta (s)</th></tr>
      </thead>
      <tbody id="leaderBody"></tbody>
    </table>

    <div id="mcSection" style="display:none;">
      <h2>🎲 Monte Carlo Win Probabilities (500 Simulated Races)</h2>
      <table>
        <thead>
          <tr><th>Strategy</th><th>Win Prob (%)</th><th>Podium (%)</th><th>Mean Time (s)</th><th>Std Dev (s)</th></tr>
        </thead>
        <tbody id="mcBody"></tbody>
      </table>
    </div>
  </main>
</div>

<script>
const COLORS = { SOFT: "#FF3B30", MEDIUM: "#FFD60A", HARD: "#E2E8F0", INTERMEDIATE: "#32D74B", WET: "#0A84FF" };

function drawDegChart(curves) {
  const c = document.getElementById("degCanvas");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const allPts = Object.values(curves).flatMap(v => v.points);
  const minY = Math.min(...allPts) - 0.2, maxY = Math.max(...allPts) + 0.2;
  const pad = { l: 42, r: 16, t: 24, b: 26 };
  const w = c.width - pad.l - pad.r, h = c.height - pad.t - pad.b;

  ctx.strokeStyle = "#1E2533"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (h * i) / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + w, y); ctx.stroke();
    ctx.fillStyle = "#94A3B8"; ctx.font = "11px sans-serif";
    ctx.fillText((maxY - ((maxY - minY) * i) / 4).toFixed(1), 6, y + 4);
  }

  let lx = pad.l;
  for (const [comp, obj] of Object.entries(curves)) {
    ctx.strokeStyle = COLORS[comp]; ctx.lineWidth = 2.5;
    ctx.beginPath();
    obj.points.forEach((val, idx) => {
      const x = pad.l + (w * idx) / (obj.points.length - 1);
      const y = pad.t + h - ((val - minY) / (maxY - minY)) * h;
      idx === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = COLORS[comp]; ctx.font = "bold 11px sans-serif";
    ctx.fillText(`${comp} (+${obj.slope}s)`, lx, 15);
    lx += 140;
  }
}

function drawPaceChart(trace) {
  const c = document.getElementById("paceCanvas");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const times = trace.map(r => r.LapTime_s);
  const minY = Math.min(...times) - 0.5, maxY = Math.max(...times) + 1.0;
  const pad = { l: 42, r: 16, t: 20, b: 26 };
  const w = c.width - pad.l - pad.r, h = c.height - pad.t - pad.b;

  ctx.strokeStyle = "#1E2533"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + (h * i) / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + w, y); ctx.stroke();
    ctx.fillStyle = "#94A3B8"; ctx.font = "11px sans-serif";
    ctx.fillText((maxY - ((maxY - minY) * i) / 4).toFixed(0), 10, y + 4);
  }

  trace.forEach((row, idx) => {
    const x = pad.l + (w * idx) / (trace.length - 1);
    const y = pad.t + h - ((row.LapTime_s - minY) / (maxY - minY)) * h;
    if (row.PitStop) {
      ctx.strokeStyle = "#E10600"; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + h); ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = COLORS[row.Compound] || "#FFF";
    ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
  });
}

async function updateSim(runMc) {
  const params = new URLSearchParams({
    total_laps: document.getElementById("totalLaps").value,
    starting_tyre: document.getElementById("startTyre").value,
    model_type: document.getElementById("modelType").value,
    base_pit_loss: document.getElementById("pitLoss").value,
    initial_weather: document.getElementById("weather").value,
    rain_prob: document.getElementById("rainProb").value,
    sc_prob: document.getElementById("scProb").value,
    run_mc: runMc
  });
  const res = await fetch(`/api/simulate?${params}`);
  const data = await res.json();

  document.getElementById("kpiSeq").innerHTML = data.best_compounds
    .map(c => `<span class="pill" style="background:${COLORS[c] || '#E2E8F0'}">${c}</span>`)
    .join(" → ");
  document.getElementById("kpiStops").textContent = `${data.best_pit_laps.length}-Stop Strategy`;
  document.getElementById("kpiWin").textContent = data.pit_windows.join(" & ");
  document.getElementById("kpiExact").textContent = `Target Box: Lap ${data.best_pit_laps.join(", ")}`;
  document.getElementById("kpiTime").textContent = data.predicted_time;
  document.getElementById("kpiDist").textContent = `${data.total_laps} Laps Total Distance`;
  document.getElementById("kpiAdv").textContent = `+${data.advantage_s.toFixed(2)} s`;
  document.getElementById("kpiRunner").textContent = `vs ${data.runner_up}`;

  const scDiv = document.getElementById("scAlert");
  scDiv.innerHTML = data.sc_advice.map(r =>
    `<div class="alert">🚨 <b>${r.track_status} on Lap ${r.lap}</b> → <b>${r.decision}</b> — ${r.reason}</div>`
  ).join("");

  drawDegChart(data.deg_curves);
  drawPaceChart(data.lap_trace);

  document.getElementById("leaderBody").innerHTML = data.leaderboard.map(r =>
    `<tr><td><b>#${r.Rank}</b></td><td>${r.Strategy}</td><td>${r.Stops}</td><td>${r.Compounds}</td><td>${r.TotalTime_s.toFixed(3)}</td><td>+${r.DeltaToBest_s.toFixed(3)}</td></tr>`
  ).join("");

  if (data.mc_summary && data.mc_summary.length > 0) {
    document.getElementById("mcSection").style.display = "block";
    document.getElementById("mcBody").innerHTML = data.mc_summary.map(r =>
      `<tr><td><b>${r.Strategy}</b></td><td>${r["WinProb_%"]}%</td><td>${r["PodiumProb_%"]}%</td><td>${r.MeanTime_s}</td><td>${r.StdTime_s}</td></tr>`
    ).join("");
  }
}

updateSim(false);
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE
