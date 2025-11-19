# app.py — Baseball Dashboard (Interactive with Plotly)

import os, glob
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Baseball Dashboard", layout="wide")
HERE = os.path.dirname(__file__)
FIELD_IMAGE = os.path.join(HERE, "baseball.png")  # no longer required, kept for compatibility
DATA_DIR = os.path.join(HERE, "data_25")
os.makedirs(DATA_DIR, exist_ok=True)

# Field extents (ft)
X_MIN, X_MAX, Y_MIN, Y_MAX = -330, 330, 0, 420
TEAM_DEFAULT = "CAL_MUS"

# ---------------- DATA LOAD (CSV-based) ----------------
def _file_signature(folder: str):
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    return tuple((p, os.path.getmtime(p), os.path.getsize(p)) for p in paths)

@st.cache_data(show_spinner=False)
def load_all_csv(sig: tuple) -> pd.DataFrame:
    csv_paths = [p for p, _, _ in sig]
    if not csv_paths:
        st.warning("No CSV files found in data_25/.")
        return pd.DataFrame()
    df = pd.concat((pd.read_csv(p) for p in csv_paths), ignore_index=True)
    return df

@st.cache_data(ttl=300, show_spinner=False)
def load_hitters(team: str) -> pd.DataFrame:
    sig = _file_signature(DATA_DIR)
    df = load_all_csv(sig)
    if df.empty or "BatterTeam" not in df.columns:
        return pd.DataFrame()
    return df[df["BatterTeam"] == team].copy()

@st.cache_data(ttl=300, show_spinner=False)
def load_pitchers(team: str) -> pd.DataFrame:
    sig = _file_signature(DATA_DIR)
    df = load_all_csv(sig)
    if df.empty or "PitcherTeam" not in df.columns:
        return pd.DataFrame()
    cols = [
        "Pitcher","PitcherThrows","PitcherTeam","AutoPitchType",
        "InducedVertBreak","HorzBreak","RelSpeed"
    ]
    present = [c for c in cols if c in df.columns]
    return df[df["PitcherTeam"] == team][present].copy() if present else pd.DataFrame()

# ---------------- HITTER HELPERS ----------------
def clean_and_xy(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    for c in ["ExitSpeed","Angle","Direction","Distance","PositionAt110X","PositionAt110Y"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    hits = df[df["ExitSpeed"].notna()].copy()
    if "PlayResult" in hits.columns:
        pr = hits["PlayResult"].astype(str).str.strip().str.lower()
        hits = hits[pr.ne("undefined")].copy()
    use_dd = hits["Distance"].notna() & hits["Direction"].notna()
    th = np.deg2rad(hits.loc[use_dd, "Direction"]); D = hits.loc[use_dd, "Distance"]
    hits.loc[use_dd, "X_ft"] = D * np.sin(th); hits.loc[use_dd, "Y_ft"] = D * np.cos(th)
    fb = hits["X_ft"].isna() & hits["PositionAt110X"].notna() & hits["PositionAt110Y"].notna()
    hits.loc[fb, "X_ft"] = hits.loc[fb, "PositionAt110X"]; hits.loc[fb, "Y_ft"] = hits.loc[fb, "PositionAt110Y"]
    return hits

# ---- drawn diamond helpers (accurate geometry) ----
BASE = 90.0
DIAG = BASE * np.sqrt(2)     # ~127.279
B = BASE / np.sqrt(2)        # ~63.64
HOME = (0.0, 0.0)
FIRST = ( B,  B)
SECOND = (0.0, DIAG)
THIRD = (-B,  B)
RUBBER_Y = 60.5              # 60'6"
MOUND_R = 9.0                # 9 ft radius

def _circle_path(cx, cy, r, n=181):
    th = np.linspace(0, 2*np.pi, n)
    x = cx + r*np.cos(th)
    y = cy + r*np.sin(th)
    return "M " + " L ".join(f"{x[i]},{y[i]}" for i in range(len(x))) + " Z"

def _arc_path(cx, cy, r, th0_deg, th1_deg, n=121):
    th = np.radians(np.linspace(th0_deg, th1_deg, n))
    x = cx + r*np.sin(th)  # your convention: x = r sin(theta), y = r cos(theta)
    y = cy + r*np.cos(th)
    return "M " + " L ".join(f"{x[i]},{y[i]}" for i in range(len(x)))

def make_field_diamond_figure(p: pd.DataFrame, batter_name: str = "Player"):
    """Draw a true-scale baseball diamond and scatter spray points on top."""
    fig = go.Figure()

    # -------- Diamond (accurate geometry) --------
    diamond_xy = [HOME, FIRST, SECOND, THIRD, HOME]
    fig.add_trace(go.Scatter(
        x=[pt[0] for pt in diamond_xy],
        y=[pt[1] for pt in diamond_xy],
        mode="lines",
        line=dict(color="#ffffff", width=3),
        showlegend=False, hoverinfo="skip"
    ))

    # Bases (15" squares)
    base_half = 0.625 / np.sqrt(2)
    for (bx, by) in (FIRST, SECOND, THIRD):
        fig.add_shape(
            type="path", xref="x", yref="y",
            path=f"M {bx},{by+base_half*2} L {bx+base_half*2},{by} "
                 f"L {bx},{by-base_half*2} L {bx-base_half*2},{by} Z",
            line=dict(color="#ffffff", width=2),
            fillcolor="#ffffff", layer="above"
        )

    # Home plate
    hp = np.array([
        [0.0, 0.0],
        [8.5/12, 0.0],
        [8.5/12, 8.5/12],
        [-8.5/12, 8.5/12],
        [-8.5/12, 0.0],
        [0.0, 0.0],
    ])
    fig.add_trace(go.Scatter(
        x=hp[:, 0], y=hp[:, 1],
        mode="lines",
        line=dict(color="#ffffff", width=3),
        showlegend=False, hoverinfo="skip"
    ))

    # Foul lines
    far = Y_MAX
    k = far / np.cos(np.radians(45))
    for sign in (-1, 1):
        fig.add_trace(go.Scatter(
            x=[0, sign * k * np.sin(np.radians(45))],
            y=[0, k * np.cos(np.radians(45))],
            mode="lines",
            line=dict(color="#ffffff", width=2),
            showlegend=False, hoverinfo="skip"
        ))

    # Pitcher’s rubber + mound
    fig.add_shape(
        type="rect", xref="x", yref="y",
        x0=-1.0, x1=1.0, y0=RUBBER_Y-0.25, y1=RUBBER_Y+0.25,
        line=dict(color="#ffffff"), fillcolor="#ffffff"
    )
    fig.add_shape(
        type="path", xref="x", yref="y",
        path=_circle_path(0.0, RUBBER_Y, MOUND_R),
        line=dict(color="#cfcfcf"), fillcolor="rgba(200,200,200,0.15)"
    )

    # Grass (rich green)
    fig.add_shape(
        type="rect", xref="x", yref="y",
        x0=X_MIN, x1=X_MAX, y0=Y_MIN, y1=Y_MAX,
        layer="below", fillcolor="#4caf50", line=dict(width=0)
    )

    # -------- Spray Points --------
    hover_cols = [c for c in ["Batter", "ExitSpeed", "Angle", "Direction", "Distance",
                              "PlayResult", "Date"] if c in p.columns]

    # Handle Exit Velocity range and ticks
    if "ExitSpeed" in p.columns and p["ExitSpeed"].notna().any():
        ev = pd.to_numeric(p["ExitSpeed"], errors="coerce")
        ev_min, ev_max = int(np.nanmin(ev)), int(np.nanmax(ev))
        ev_min = (ev_min // 10) * 10
        ev_max = ((ev_max + 9) // 10) * 10
        tickvals = list(range(ev_min, ev_max + 1, 10))
    else:
        ev = None
        tickvals = None

    fig.add_trace(go.Scatter(
        x=p["X_ft"], y=p["Y_ft"],
        mode="markers",
        marker=dict(
            size=9,
            color=ev if ev is not None else None,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(
                title="<b>Exit Velocity (MPH)</b>",
                tickvals=tickvals,
                tickformat=".0f",
                ticks="outside",
                tickfont=dict(size=11, color="#333"),
                titlefont=dict(size=12, color="#333"),
                thickness=18,
                outlinewidth=0,
                yanchor="middle",
                y=0.5,
            ),
            opacity=0.9,
        ),
        customdata=p[hover_cols].values if hover_cols else None,
        hovertemplate="<br>".join(
            [f"<b>{c}</b>: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols)]
        ) if hover_cols else None,
        showlegend=False,
        name=""
    ))

    # -------- Layout --------
    fig.update_xaxes(range=[X_MIN, X_MAX], title_text="Horizontal (ft)  [− = LF, + = RF]",
                     showgrid=False, zeroline=False)
    fig.update_yaxes(range=[Y_MIN, Y_MAX], title_text="Depth (ft)  [0 = Home, + = CF]",
                     scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False)

    fig.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(
            text=f"{batter_name} : Batting Spray Chart",
            x=0.5, xanchor="center",
            font=dict(size=20, color="#222")
        ),
        showlegend=False
    )
    return fig




def make_polar_spray_figure(req: pd.DataFrame):
    """Interactive polar-like spray using Direction/Distance in Cartesian with angle hover."""
    if req.empty:
        return go.Figure().update_layout(title="No Direction/Distance available")

    th = np.deg2rad(req["Direction"].to_numpy())
    r  = req["Distance"].to_numpy()
    x  = r * np.sin(th)
    y  = r * np.cos(th)

    hover_cols = []
    for c in ["Batter","ExitSpeed","Angle","Direction","Distance","PlayResult","Date"]:
        if c in req.columns: hover_cols.append(c)

    fig = px.scatter(
        x=x, y=y,
        color=req["ExitSpeed"] if "ExitSpeed" in req.columns else None,
        color_continuous_scale="Plasma",
        labels={"x":"Horizontal (ft)  [− = LF, + = RF]", "y":"Depth (ft)  [0 = Home, + = CF]"},
    )
    fig.update_traces(
        marker=dict(size=9, opacity=0.9),
        customdata=req[hover_cols].values if hover_cols else None,
        hovertemplate="<br>".join([f"<b>{c}</b>: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols)]) if hover_cols else None,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    return fig

# ---------------- PITCHER HELPERS ----------------
PITCH_COLORS = {
    "Four-Seam": "#d9534f",
    "Sinker":    "#f0ad4e",
    "Cutter":    "#b294bb",
    "Changeup":  "#88d498",
    "Splitter":  "#8dd3c7",
    "Slider":    "#f0e442",
    "Curveball": "#5bc0de",
    "Other":     "#7f8c8d",
}

def prep_pitcher_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    for c in ["InducedVertBreak","HorzBreak","RelSpeed"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    valid = {"Four-Seam","Sinker","Cutter","Changeup","Splitter","Slider","Curveball","Other"}
    df = df[
        df["Pitcher"].notna() &
        df["AutoPitchType"].astype(str).isin(valid) &
        df["InducedVertBreak"].notna() &
        df["HorzBreak"].notna() &
        df["RelSpeed"].notna()
    ].copy()
    return df

def make_movement_figure(sub: pd.DataFrame, normalize_lhp: bool, throws_hand: str):
    data = sub.copy()
    if normalize_lhp and str(throws_hand).upper().startswith("L"):
        data["HorzBreak"] = -data["HorzBreak"]

    hover_cols = ["AutoPitchType","RelSpeed","InducedVertBreak","HorzBreak"]
    present = [c for c in hover_cols if c in data.columns]

    fig = px.scatter(
        data, x="HorzBreak", y="InducedVertBreak",
        color="AutoPitchType",
        color_discrete_map=PITCH_COLORS,
        hover_data=present,
        labels={"HorzBreak": 'Horizontal Break (")', "InducedVertBreak": 'Induced Vertical Break (")'},
    )

    for r in (12, 24):
        fig.add_shape(type="circle", xref="x", yref="y",
                      x0=-r, y0=-r, x1=r, y1=r,
                      line=dict(color="#cfd8dc", width=1.5))

    fig.add_shape(type="line", x0=-100, x1=100, y0=0, y1=0, line=dict(color="#b0bec5", width=1))
    fig.add_shape(type="line", x0=0, x1=0, y0=-100, y1=100, line=dict(color="#b0bec5", width=1))

    max_abs = float(np.nanmax(np.abs(data[["HorzBreak","InducedVertBreak"]])))
    lim = int(np.ceil(max(24, max_abs + 4) / 6.0) * 6)
    lim = min(lim, 36)
    fig.update_xaxes(range=[-lim, lim], tickvals=[-24,-12,0,12,24], constrain="domain")
    fig.update_yaxes(range=[-lim, lim], tickvals=[-24,-12,0,12,24], scaleanchor="x", scaleratio=1)

    fig.update_layout(legend_title_text="Pitch", margin=dict(l=10,r=10,t=40,b=10))
    return fig

# ---------------- UI ----------------
st.title("Baseball Dashboard")
team = st.sidebar.text_input("Team", TEAM_DEFAULT)
section = st.sidebar.radio("Section", ["Hitter Dashboard", "Pitcher Dashboard"])

# ---- Data Management (sidebar) ----
st.sidebar.subheader("Data Management")
new_files = st.sidebar.file_uploader("Add CSVs", type=["csv"], accept_multiple_files=True)
if new_files:
    for f in new_files:
        with open(os.path.join(DATA_DIR, f.name), "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success(f"Saved {len(new_files)} file(s) to data_25/")
    st.cache_data.clear()

if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.sidebar.info("Cache cleared. Data will reload on next run.")

# ===== HITTER DASHBOARD =====
if section == "Hitter Dashboard":
    raw = load_hitters(team); df = clean_and_xy(raw)
    batters_all = sorted(df["Batter"].dropna().unique())
    if not batters_all:
        st.warning("No batters with batted-ball data for this team."); st.stop()

    mode = st.sidebar.radio("Mode", ["Single Batter", "Compare Batters"])

    if mode == "Single Batter":
        batter = st.sidebar.selectbox("Batter", batters_all)
        s_all = df[df["Batter"] == batter].dropna(subset=["X_ft","Y_ft"]).copy()

        # KPIs
        ev_avg = s_all["ExitSpeed"].mean()
        ev_max = s_all["ExitSpeed"].max()
        la_avg = s_all["Angle"].mean() if ("Angle" in s_all.columns and s_all["Angle"].notna().any()) else np.nan
        hh_rate = (s_all["ExitSpeed"] >= 95).mean() if len(s_all) else np.nan

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("BIP", len(s_all))
        c2.metric("Avg EV", f"{ev_avg:.1f} mph" if len(s_all) else "–")
        c3.metric("Max EV", f"{ev_max:.1f} mph" if len(s_all) else "–")
        c4.metric("Avg LA", f"{la_avg:.1f}°" if len(s_all) and pd.notna(la_avg) else "–")
        c5.metric("Hard-Hit %", f"{100*hh_rate:.0f}%" if len(s_all) else "–")

        chart_type = st.radio("Chart Type", ["Field Overlay", "Polar"], horizontal=True)

        if chart_type == "Field Overlay":
            fig = make_field_diamond_figure(s_all, batter_name=batter)  # <<—— draw diamond + spray
        else:
            req = s_all.dropna(subset=["Direction","Distance"])
            fig = make_polar_spray_figure(req)
        st.plotly_chart(fig, use_container_width=True, theme=None)

        st.subheader("Batted Balls (all BIP)")
        cols = ["Date","Batter","ExitSpeed","Angle","Direction","Distance","PlayResult","X_ft","Y_ft"]
        show = [c for c in cols if c in s_all.columns]
        st.dataframe(s_all[show].sort_values("Date", ascending=False), use_container_width=True)

    else:
        chosen = st.sidebar.multiselect("Choose up to 12 batters", batters_all, default=batters_all[:10], max_selections=12)
        if len(chosen) == 0:
            st.info("Select at least one batter."); st.stop()

        plot_df = df[df["Batter"].isin(chosen)].dropna(subset=["X_ft","Y_ft"]).copy()
        if plot_df.empty:
            st.info("No XY for selected batters.")
        else:
            hover_cols = [c for c in ["Batter","ExitSpeed","Angle","Direction","Distance","PlayResult","Date"] if c in plot_df.columns]
            fig = px.scatter(
                plot_df, x="X_ft", y="Y_ft",
                color="ExitSpeed" if "ExitSpeed" in plot_df.columns else None,
                color_continuous_scale="Plasma",
                facet_col="Batter", facet_col_wrap=4,
                hover_data=hover_cols,
                labels={"X_ft":"Horizontal (ft)", "Y_ft":"Depth (ft)"},
            )
            fig.update_traces(marker=dict(size=6, opacity=0.9))
            fig.for_each_xaxis(lambda ax: ax.update(range=[X_MIN, X_MAX], showgrid=False))
            fig.for_each_yaxis(lambda ay: ay.update(range=[Y_MIN, Y_MAX], showgrid=False, scaleanchor="x", scaleratio=1))
            fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
            st.plotly_chart(fig, use_container_width=True, theme=None)

# ===== PITCHER DASHBOARD =====
else:
    rawp = load_pitchers(team); dpp = prep_pitcher_df(rawp)
    pitchers = sorted(dpp["Pitcher"].unique().tolist())
    if not pitchers:
        st.warning("No pitcher data with movement/velo for this team."); st.stop()

    pitcher = st.sidebar.selectbox("Pitcher", pitchers)
    sub = dpp[dpp["Pitcher"] == pitcher].copy()

    throws = (sub["PitcherThrows"].dropna().iloc[0] if sub["PitcherThrows"].notna().any() else "R").upper()
    normalize = st.sidebar.checkbox("Normalize LHP to RHP frame (+HB = arm-side)", value=True)

    fig = make_movement_figure(sub, normalize_lhp=normalize, throws_hand=throws)
    st.subheader(f"Movement Profile — {pitcher}  (Throws: {throws})")
    st.plotly_chart(fig, use_container_width=True, theme=None)

    summary = (
        sub.groupby("AutoPitchType")
           .agg(Count=("AutoPitchType","size"), MPH=("RelSpeed","mean"))
           .sort_values("Count", ascending=False)
           .reset_index()
    )
    summary["Usage %"] = (100 * summary["Count"] / max(1, summary["Count"].sum())).round(0).astype(int)
    summary["MPH"] = summary["MPH"].round(1)
    st.subheader("Pitch Mix & Velo")
    st.dataframe(summary, use_container_width=True)
