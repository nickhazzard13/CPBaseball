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

st.markdown("""
<style>
/* Make EVERYTHING dark */
html, body, .stApp {
    background-color: #000000 !important;
}

/* Top header bar */
header[data-testid="stHeader"],
header[data-testid="stHeader"] > div {
    background-color: #000000 !important;
    box-shadow: none !important;
}

/* Remove the white band by also darkening main containers */
.main, .block-container {
    background-color: #000000 !important;
    padding-top: 0.5rem !important;
}

/* === SIDEBAR: dark grey strip on the left === */

/* Full-height sidebar column */
section[data-testid="stSidebar"] {
    background-color: #1e1e1e !important;   /* dark grey */
}

/* Inner sidebar content */
section[data-testid="stSidebar"] > div:first-child {
    background-color: #1e1e1e !important;   /* dark grey */
    padding-top: 0.75rem !important;
    padding-bottom: 1rem !important;
    border-right: 1px solid #333333;       /* subtle divider line */
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    font-size: 0.9rem;
}

/* Sidebar inputs (Team textbox, selectboxes, etc.) */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] select {
    background-color: #2a2a2a !important;   /* slightly lighter than panel */
    color: #FFFFFF !important;
    border-radius: 6px !important;
    border: 1px solid #444444 !important;
}

/* Sidebar radio + checkbox labels */
section[data-testid="stSidebar"] label[data-baseweb="radio"],
section[data-testid="stSidebar"] label[data-baseweb="checkbox"] {
    color: #FFFFFF !important;
}

/* File uploader in sidebar */
section[data-testid="stSidebar"] [data-testid="stFileUploader"] div[data-baseweb="base-input"],
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background-color: #2a2a2a !important;
    border-radius: 8px !important;
    border: 1px dashed #444444 !important;
}

/* Buttons (Refresh, etc.) */
.stButton>button {
    border-radius: 999px !important;
    border: 1px solid #FFE395 !important;
    background-color: #111111 !important;
    color: #FFE395 !important;
}
.stButton>button:hover {
    background-color: #FFE395 !important;
    color: #000000 !important;
}

/* Headings: gold */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: #FFE395 !important;
}

/* Tab bar accent */
button[data-baseweb="tab"] {
    color: #FFFFFF !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #FFE395 !important;
    border-bottom: 2px solid #FFE395 !important;
}

/* Metrics labels + values */
[data-testid="stMetricLabel"] {
    font-size: 0.8rem;
    color: #CCCCCC !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.7rem;
    font-weight: 600;
    color: #FFFFFF !important;
}
.block-container {
    padding-top: 2.5rem !important;
}
div[role="radiogroup"] > label {
    color: #EEEEEE !important;
    font-weight: 500 !important;
}

/* Lighten the text next to the bullet */
div[role="radiogroup"] span {
    color: #EEEEEE !important;
}
div[role="radiogroup"] label span {
    color: #FFFFFF !important;
}

/* Make the radio circles also white when unselected */
div[role="radiogroup"] input[type="radio"] {
    accent-color: #FFFFFF !important;
}
div[role="radiogroup"] label > div:nth-child(2) {
    color: #FFFFFF !important;
}

/* Also handle Streamlit’s internal span wrapper */
div[role="radiogroup"] label span {
    color: #FFFFFF !important;
}
</style>
""", unsafe_allow_html=True)
HERE = os.path.dirname(__file__)
FIELD_IMAGE = os.path.join(HERE, "baseball.png")  # no longer required, kept for compatibility
DATA_DIR = os.path.join(HERE, "data_25")
os.makedirs(DATA_DIR, exist_ok=True)

# Field extents (ft)
X_MIN, X_MAX, Y_MIN, Y_MAX = -330, 330, 0, 420
TEAM_DEFAULT = "CAL_MUS"

# =========================================================
# 0. COMMON DATA LOAD (CSV-based)
# =========================================================

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

    df = df[df["BatterTeam"] == team].copy()

    # --- CLEAN BATTER NAMES ---
    if "Batter" in df.columns:
        df["Batter"] = df["Batter"].apply(_standardize_name)

    return df


@st.cache_data(ttl=300, show_spinner=False)
def load_pitchers(team: str) -> pd.DataFrame:
    """
    Load all pitcher rows for a team with full columns
    (we need plate location + EV for zone heatmaps and movement columns for plots).
    """
    sig = _file_signature(DATA_DIR)
    df = load_all_csv(sig)
    if df.empty or "PitcherTeam" not in df.columns:
        return pd.DataFrame()
    return df[df["PitcherTeam"] == team].copy()


def _standardize_name(s: str) -> str:
    """
    Strip spaces, collapse internal whitespace, and title-case the name.
    This will merge 'Nick Bonn', 'nick bonn ', 'NICK  BONN', 'Coco VonderHaar', etc.
    """
    if pd.isna(s):
        return s
    s = str(s).strip()
    s = " ".join(s.split())  # collapse multiple spaces
    s = s.title()            # 'coco vonderhaar' -> 'Coco Vonderhaar'
    return s

# =========================================================
# 1. ZONE / HEATMAP ENGINE (HITTERS & PITCHERS)
# =========================================================

def normalize_plate_coords(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["PlateLocHeight", "PlateLocSide"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # Drop rows missing plate location only
    out = out.dropna(subset=["PlateLocHeight", "PlateLocSide"])
    # No handedness normalization; keep absolute zone
    out["PlateLocSide_norm"] = out["PlateLocSide"]
    return out

def assign_zone_bins(df: pd.DataFrame,
                     side_bounds=(-0.9, 0.9),
                     height_bounds=(1.5, 3.9),
                     side_cut=0.3,
                     low_cut=2.3,
                     high_cut=3.1):
    """
    Adds zone_row, zone_col, zone_id.
    Uses PlateLocSide_norm (which is just PlateLocSide now).
    """
    out = df.copy()
    if "PlateLocHeight" not in out.columns or "PlateLocSide_norm" not in out.columns:
        return out

    # Filter to zone-ish region
    h_min, h_max = height_bounds
    s_min, s_max = side_bounds

    mask = (out["PlateLocHeight"].between(h_min, h_max) &
            out["PlateLocSide_norm"].between(s_min, s_max))
    out = out[mask].copy()

    # Vertical bins (0 = low, 1 = mid, 2 = high)
    h = out["PlateLocHeight"]
    r = pd.Series(np.nan, index=out.index)
    r[h < low_cut] = 0
    r[(h >= low_cut) & (h < high_cut)] = 1
    r[h >= high_cut] = 2

    # Horizontal bins (0 = away/left, 1 = middle, 2 = in/right)
    s = out["PlateLocSide_norm"]
    c = pd.Series(np.nan, index=out.index)
    c[s < -side_cut] = 0
    c[(s >= -side_cut) & (s <= side_cut)] = 1
    c[s > side_cut] = 2

    out["zone_row"] = r.astype("Int64")
    out["zone_col"] = c.astype("Int64")

    # zone_id = row * 3 + col
    out["zone_id"] = np.where(
        out["zone_row"].notna() & out["zone_col"].notna(),
        out["zone_row"].astype(int) * 3 + out["zone_col"].astype(int),
        pd.NA
    )
    return out

SWING_CALLS = {
    "StrikeSwinging", "StrikeSwingingOnFoulTip",
    "InPlay", "FoulBall", "FoulBallNotFieldable",
    "FoulBallFieldable", "FoulTip"
}
INPLAY_CALLS = {"InPlay"}
CALLED_STRIKE_CALLS = {"StrikeCalled", "StrikeLooking"}

def add_swing_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "PitchCall" not in out.columns:
        return out

    pc = out["PitchCall"].astype(str)
    pr = out["PlayResult"].astype(str) if "PlayResult" in out.columns else ""

    # Swing / whiff / contact
    out["is_swing"] = pc.isin(SWING_CALLS)
    out["is_whiff"] = pc.eq("StrikeSwinging")
    out["is_contact"] = out["is_swing"] & ~out["is_whiff"]
    out["is_bip"] = pc.isin(INPLAY_CALLS)
    out["is_called_strike"] = pc.isin(CALLED_STRIKE_CALLS)

    # Hits & bases
    hit_vals = {"Single": 1, "Double": 2, "Triple": 3, "HomeRun": 4}
    if "PlayResult" in out.columns:
        out["is_hit"] = pr.isin(hit_vals.keys())
        out["bases"] = out["PlayResult"].map(hit_vals).fillna(0)
    else:
        out["is_hit"] = False
        out["bases"] = 0

    # EV & LA
    if "ExitSpeed" in out.columns:
        out["ExitSpeed"] = pd.to_numeric(out["ExitSpeed"], errors="coerce")
    else:
        out["ExitSpeed"] = np.nan

    if "Angle" in out.columns:
        out["Angle"] = pd.to_numeric(out["Angle"], errors="coerce")
    else:
        out["Angle"] = np.nan

    # Hard-hit: EV >= 90 on BIP
    out["is_hard_hit"] = out["is_bip"] & (out["ExitSpeed"] >= 90)

    # Sweet spot: 8° <= LA <= 32° on BIP
    out["is_sweet_spot"] = out["is_bip"] & out["Angle"].between(8, 32)

    return out

def compute_hitter_zone_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-batter, per-zone damage score.
    """
    if not {"Batter", "zone_row", "zone_col"}.issubset(df.columns):
        return pd.DataFrame()

    g = df.groupby(["Batter", "zone_row", "zone_col"], dropna=True)

    agg = g.agg(
        pitches=("PitchCall", "size"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        contacts=("is_contact", "sum"),
        bip=("is_bip", "sum"),
        ev_sum=("ExitSpeed", "sum"),
        ev_n=("ExitSpeed", lambda x: x.notna().sum()),
        hard_hits=("is_hard_hit", "sum"),
        sweet_spots=("is_sweet_spot", "sum"),
        la_n=("Angle", lambda x: x.notna().sum()),
    ).reset_index()

    # Rates
    agg["ZoneContact%"] = np.where(agg["swings"] > 0, agg["contacts"] / agg["swings"], 0.0)
    agg["Whiff%"] = np.where(agg["swings"] > 0, agg["whiffs"] / agg["swings"], 0.0)
    agg["EV_avg"] = np.where(agg["ev_n"] > 0, agg["ev_sum"] / agg["ev_n"], 0.0)
    agg["HardHit%"] = np.where(agg["bip"] > 0, agg["hard_hits"] / agg["bip"], 0.0)
    agg["SweetSpot%"] = np.where(agg["bip"] > 0, agg["sweet_spots"] / agg["bip"], 0.0)

    # xwOBA-style
    agg["xwOBAcon"] = (
        0.0025 * agg["EV_avg"] +
        0.35   * agg["SweetSpot%"] +
        0.35   * agg["HardHit%"]
    )
    agg["xwOBA"] = agg["xwOBAcon"] * (1 - 0.5 * agg["Whiff%"])

    # Subscores (0–100)
    agg["sub_xwOBA"]     = 100 * agg["xwOBA"].clip(0, 1)
    agg["sub_xwOBAcon"]  = 100 * agg["xwOBAcon"].clip(0, 1)
    agg["sub_EV"]        = 100 * (agg["EV_avg"] / 110).clip(0, 1)
    agg["sub_SweetSpot"] = 100 * agg["SweetSpot%"].clip(0, 1)
    agg["sub_HardHit"]   = 100 * agg["HardHit%"].clip(0, 1)
    agg["sub_Contact"]   = 100 * agg["ZoneContact%"].clip(0, 1)
    agg["sub_Whiff"]     = 100 * agg["Whiff%"].clip(0, 1)

    # Composite score (you can tweak these later)
    agg["score_raw"] = (
        0.28 * agg["sub_xwOBA"] +
        0.12 * agg["sub_xwOBAcon"] +
        0.20 * agg["sub_EV"] +
        0.10 * agg["sub_SweetSpot"] +
        0.12 * agg["sub_HardHit"] +
        0.08 * agg["sub_Contact"] +
        0.10 * (100 - agg["sub_Whiff"])
    )

    # Sample-size shrinkage toward 50 (swings control reliability)
    agg["sample_weight"] = np.minimum(1.0, agg["swings"] / 10.0)
    agg["HitterZoneScore"] = (
        50.0 * (1 - agg["sample_weight"]) +
        agg["score_raw"] * agg["sample_weight"]
    ).clip(0, 100)

    return agg

def compute_pitcher_zone_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-pitcher, per-zone dominance/prevention score.
    """
    if not {"Pitcher", "zone_row", "zone_col"}.issubset(df.columns):
        return pd.DataFrame()

    g = df.groupby(["Pitcher", "zone_row", "zone_col"], dropna=True)

    agg = g.agg(
        pitches=("PitchCall", "size"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        contacts=("is_contact", "sum"),
        bip=("is_bip", "sum"),
        called_strikes=("is_called_strike", "sum"),
        ev_sum=("ExitSpeed", "sum"),
        ev_n=("ExitSpeed", lambda x: x.notna().sum()),
        hard_hits=("is_hard_hit", "sum"),
        sweet_spots=("is_sweet_spot", "sum"),
        la_n=("Angle", lambda x: x.notna().sum()),
    ).reset_index()

    # Rates
    agg["CSW%"] = np.where(
        agg["pitches"] > 0,
        (agg["called_strikes"] + agg["whiffs"]) / agg["pitches"],
        0.0
    )
    agg["Whiff%"] = np.where(
        agg["swings"] > 0,
        agg["whiffs"] / agg["swings"],
        0.0
    )
    agg["EV_allowed"] = np.where(
        agg["ev_n"] > 0,
        agg["ev_sum"] / agg["ev_n"],
        0.0
    )
    agg["HardHit%_allowed"] = np.where(
        agg["bip"] > 0,
        agg["hard_hits"] / agg["bip"],
        0.0
    )
    agg["SweetSpot%_allowed"] = np.where(
        agg["bip"] > 0,
        agg["sweet_spots"] / agg["bip"],
        0.0
    )

    agg["xwOBAcon_allowed"] = (
        0.0025 * agg["EV_allowed"] +
        0.35   * agg["SweetSpot%_allowed"] +
        0.35   * agg["HardHit%_allowed"]
    )
    agg["xwOBA_allowed"] = agg["xwOBAcon_allowed"] * (1 - 0.5 * agg["Whiff%"])

    # Subscores (0–100), oriented so HIGH = GOOD for pitchers
    agg["sub_CSW"]        = 100 * agg["CSW%"].clip(0, 1)
    agg["sub_Whiff"]      = 100 * agg["Whiff%"].clip(0, 1)
    agg["sub_EV_allowed"] = 100 * (1 - (agg["EV_allowed"] / 110).clip(0, 1))
    agg["sub_HardHit_a"]  = 100 * (1 - agg["HardHit%_allowed"].clip(0, 1))
    agg["sub_SweetSpot_a"]= 100 * (1 - agg["SweetSpot%_allowed"].clip(0, 1))
    agg["sub_xwOBAcon_a"] = 100 * (1 - agg["xwOBAcon_allowed"].clip(0, 1))
    agg["sub_xwOBA_a"]    = 100 * (1 - agg["xwOBA_allowed"].clip(0, 1))

    # Composite score (weights sum to 1.0)
    agg["score_raw"] = (
        0.25 * agg["sub_xwOBA_a"] +
        0.10 * agg["sub_xwOBAcon_a"] +
        0.10 * agg["sub_EV_allowed"] +
        0.15 * agg["sub_HardHit_a"] +
        0.12 * agg["sub_SweetSpot_a"] +
        0.15 * agg["sub_CSW"] +
        0.13 * agg["sub_Whiff"]
    )

    # Sample-size shrinkage toward 50 (pitches control reliability)
    agg["sample_weight"] = np.minimum(1.0, agg["pitches"] / 10.0)
    agg["PitcherZoneScore"] = (
        50.0 * (1 - agg["sample_weight"]) +
        agg["score_raw"] * agg["sample_weight"]
    ).clip(0, 100)

    return agg

def get_hitter_zone_matrix(hitter_stats: pd.DataFrame, batter_name: str) -> pd.DataFrame:
    sub = hitter_stats[hitter_stats["Batter"] == batter_name].copy()
    if sub.empty:
        raise ValueError(f"No zone data found for batter: {batter_name}")
    mat = sub.pivot(index="zone_row", columns="zone_col", values="HitterZoneScore")
    return mat.reindex(index=[0, 1, 2], columns=[0, 1, 2])

def get_pitcher_zone_matrix(pitcher_stats: pd.DataFrame, pitcher_name: str) -> pd.DataFrame:
    sub = pitcher_stats[pitcher_stats["Pitcher"] == pitcher_name].copy()
    if sub.empty:
        raise ValueError(f"No zone data found for pitcher: {pitcher_name}")
    mat = sub.pivot(index="zone_row", columns="zone_col", values="PitcherZoneScore")
    return mat.reindex(index=[0, 1, 2], columns=[0, 1, 2])

@st.cache_data(ttl=300, show_spinner=False)
def team_hitter_zone_stats(team: str) -> pd.DataFrame:
    raw = load_hitters(team)
    if raw.empty:
        return pd.DataFrame()
    df = normalize_plate_coords(raw)
    df = assign_zone_bins(df)
    df = add_swing_flags(df)
    return compute_hitter_zone_stats(df)

@st.cache_data(ttl=300, show_spinner=False)
def team_pitcher_zone_stats(team: str) -> pd.DataFrame:
    raw = load_pitchers(team)
    if raw.empty:
        return pd.DataFrame()
    df = normalize_plate_coords(raw)
    df = assign_zone_bins(df)
    df = add_swing_flags(df)
    return compute_pitcher_zone_stats(df)

def make_zone_heatmap(mat: pd.DataFrame, title: str) -> go.Figure:
    """
    Build a 3x3 heatmap like the MLB zone graphic.
    """
    z = mat.to_numpy(dtype=float)
    # Flip vertically so row 2 plots at top
    z_plot = z[::-1, :]

    # Labels
    x_labels = ["Left", "Middle", "Right"]
    y_labels = ["High", "Middle", "Low"]

    # Text labels in cells
    text_vals = np.round(z_plot, 1)
    text = np.where(np.isnan(text_vals), "", text_vals.astype(str))

    fig = go.Figure(
        data=go.Heatmap(
            z=z_plot,
            x=x_labels,
            y=y_labels,
            colorscale="Plasma",
            reversescale=True,
            zmin=0, zmax=100,
            colorbar=dict(title="Score"),
            hovertemplate="Vert: %{y}<br>Horiz: %{x}<br>Score: %{z:.1f}<extra></extra>",
        )
    )

    fig.update_traces(
        text=text,
        texttemplate="%{text}",
        textfont=dict(color="white", size=16)
    )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Horizontal Location (catcher view: Left = 3B, Right = 1B)"),
        yaxis=dict(title="Vertical Location"),
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
    )
    return fig

# ------- NEW: shared helpers for pitch types, count labels, and zone scatter -------

def standardize_pitch_type(df: pd.DataFrame,
                           source_col: str = "AutoPitchType",
                           target_col: str = "PitchTypeDisplay") -> pd.DataFrame:
    """
    Map any unknown pitch types into 'Other' so they still get a color.
    """
    out = df.copy()
    if source_col not in out.columns:
        out[target_col] = "Other"
        return out
    out[target_col] = out[source_col].astype(str)
    valid = set(PITCH_COLORS.keys())
    mask = ~out[target_col].isin(valid)
    out.loc[mask, target_col] = "Other"
    return out

def add_count_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'CountStr' column like '0-0', '1-2' from Balls/Strikes, when available.
    """
    out = df.copy()
    if {"Balls", "Strikes"}.issubset(out.columns):
        out["Balls"] = pd.to_numeric(out["Balls"], errors="coerce")
        out["Strikes"] = pd.to_numeric(out["Strikes"], errors="coerce")
        mask = out["Balls"].notna() & out["Strikes"].notna()
        out.loc[mask, "CountStr"] = (
            out.loc[mask, "Balls"].astype(int).astype(str)
            + "-"
            + out.loc[mask, "Strikes"].astype(int).astype(str)
        )
    else:
        out["CountStr"] = np.nan
    return out

def add_zone_grid_to_fig(
    fig: go.Figure,
    side_bounds=(-0.85, 0.85),
    height_bounds=(1.5, 3.5),
    side_cut=0.3,
    low_cut=2.3,
    high_cut=3.1,
):
    """
    Draw a 3x3 strike-zone grid on a PlateLocSide_norm vs PlateLocHeight scatter plot.
    Catcher view: negative side = 3B side, positive = 1B side.
    """
    s_min, s_max = side_bounds
    h_min, h_max = height_bounds

    # Outer rectangle for the zone
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=s_min,
        x1=s_max,
        y0=h_min,
        y1=h_max,
        line=dict(color="#dddddd", width=2),
        fillcolor="rgba(0,0,0,0)",
        layer="below",
    )

    # Vertical cuts (left / middle / right)
    for x_line in (-side_cut, side_cut):
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x_line,
            x1=x_line,
            y0=h_min,
            y1=h_max,
            line=dict(color="#aaaaaa", width=1, dash="solid"),
            layer="below",
        )

    # Horizontal cuts (low / mid / high)
    for y_line in (low_cut, high_cut):
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=s_min,
            x1=s_max,
            y0=y_line,
            y1=y_line,
            line=dict(color="#aaaaaa", width=1, dash="solid"),
            layer="below",
        )

    # Tight ranges around zone with a little padding
    pad_x = 0.15
    pad_y = 0.25
    fig.update_xaxes(range=[s_min - pad_x, s_max + pad_x])
    fig.update_yaxes(range=[h_min - pad_y, h_max + pad_y])

    # Make the zone rectangular (no distortion)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

def make_hitter_bip_zone_figure(df: pd.DataFrame, batter_name: str) -> go.Figure:
    """
    Scatter of BIP filtered by EV/LA in the strike-zone plane, colored by pitch type.
    """
    hover_cols = [
        c
        for c in [
            "PitchTypeDisplay",
            "AutoPitchType",
            "ExitSpeed",
            "Angle",
            "PlayResult",
            "Date",
            "CountStr",
        ]
        if c in df.columns
    ]
    color_col = "PitchTypeDisplay" if "PitchTypeDisplay" in df.columns else (
        "AutoPitchType" if "AutoPitchType" in df.columns else None
    )

    fig = px.scatter(
        df,
        x="PlateLocSide_norm",
        y="PlateLocHeight",
        color=color_col,
        color_discrete_map=PITCH_COLORS if color_col else None,
        hover_data=hover_cols,
        labels={
            "PlateLocSide_norm": "Horizontal Location (catcher view: Left = 3B, Right = 1B)",
            "PlateLocHeight": "Vertical Location (ft)",
        },
    )

    # ---- make pitch markers larger with a white outline ----
    fig.update_traces(
        marker=dict(
            size=14,
            opacity=0.95,
            line=dict(color="white", width=1.5),
        )
    )

    add_zone_grid_to_fig(fig)
    fig.update_layout(
        title=f"{batter_name} — BIP EV/LA Filtered Zone View",
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title_text="Pitch Type",
    )
    return fig


def make_pitcher_count_zone_figure(df: pd.DataFrame, pitcher_name: str, count_label: str) -> go.Figure:
    """
    Scatter of pitch locations in the zone plane for a given count (or all),
    colored by pitch type. Filled markers indicate pitches that resulted in hits.
    """
    # Flag pitches that resulted in hits
    hit_vals = {"Single", "Double", "Triple", "HomeRun"}
    if "PlayResult" in df.columns:
        df = df.copy()
        df["is_hit_pitch"] = df["PlayResult"].astype(str).isin(hit_vals)
    else:
        df = df.copy()
        df["is_hit_pitch"] = False

    hover_cols = [
        c
        for c in [
            "PitchTypeDisplay",
            "AutoPitchType",
            "RelSpeed",
            "PitchCall",
            "PlayResult",
            "Date",
            "CountStr",
            "is_hit_pitch",
        ]
        if c in df.columns
    ]

    color_col = "PitchTypeDisplay" if "PitchTypeDisplay" in df.columns else (
        "AutoPitchType" if "AutoPitchType" in df.columns else None
    )

    title = f"{pitcher_name} — Pitch Locations by Count"
    if count_label and count_label != "All":
        title += f" (Count: {count_label})"

    fig = px.scatter(
        df,
        x="PlateLocSide_norm",
        y="PlateLocHeight",
        color=color_col,
        color_discrete_map=PITCH_COLORS if color_col else None,
        symbol="is_hit_pitch",                      # True/False -> different marker shapes
        symbol_map={True: "circle", False: "circle-open"},
        hover_data=hover_cols,
        labels={
            "PlateLocSide_norm": "Horizontal Location (catcher view: Left = 3B, Right = 1B)",
            "PlateLocHeight": "Vertical Location (ft)",
            "is_hit_pitch": "Hit?",
        },
    )

    # ---- bigger pitch markers with outline ----
    fig.update_traces(
        marker=dict(
            size=14,
            opacity=0.95,
            line=dict(color="white", width=1.5),
        )
    )

    add_zone_grid_to_fig(fig)
    fig.update_layout(
        title=title,
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title_text="Pitch Type",
    )
    return fig


# =========================================================
# 2. HITTER SPRAY CHART HELPERS
# =========================================================

def clean_and_xy(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    for c in ["ExitSpeed","Angle","Direction","Distance","PositionAt110X","PositionAt110Y"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "ExitSpeed" not in df.columns:
        return pd.DataFrame()
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

def make_field_diamond_figure(p: pd.DataFrame, batter_name: str = "Player"):
    """Draw a true-scale baseball diamond and scatter spray points on top."""
    fig = go.Figure()

    # -------- Diamond --------
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

    # Grass
    fig.add_shape(
        type="rect", xref="x", yref="y",
        x0=X_MIN, x1=X_MAX, y0=Y_MIN, y1=Y_MAX,
        layer="below", fillcolor="#4caf50", line=dict(width=0)
    )

    # -------- Spray Points --------
    hover_cols = [c for c in ["Batter", "ExitSpeed", "Angle", "Direction", "Distance",
                              "PlayResult", "Date"] if c in p.columns]

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
            reversescale=True,
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

    hover_cols = [c for c in ["Batter","ExitSpeed","Angle","Direction","Distance","PlayResult","Date"]
                  if c in req.columns]

    fig = px.scatter(
        x=x, y=y,
        color=req["ExitSpeed"] if "ExitSpeed" in req.columns else None,
        color_continuous_scale="Plasma",
        labels={"x":"Horizontal (ft)  [− = LF, + = RF]", "y":"Depth (ft)  [0 = Home, + = CF]"},
    )
    fig.update_traces(
        marker=dict(size=9, opacity=0.9),
        customdata=req[hover_cols].values if hover_cols else None,
        hovertemplate="<br>".join(
            [f"<b>{c}</b>: %{{customdata[{i}]}}" for i, c in enumerate(hover_cols)]
        ) if hover_cols else None,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    return fig

# =========================================================
# 3. PITCHER MOVEMENT HELPERS
# =========================================================

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
        if c in df.columns:
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
# ==============GRAPH EDITING ===========================================
def style_dark_plotly(fig: go.Figure) -> go.Figure:
    """Apply a consistent dark style to Plotly figures."""
    fig.update_layout(
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font=dict(color="#F5F5F5"),
        margin=dict(l=10, r=10, t=60, b=30),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#333333",
        zeroline=False, linecolor="#666666", tickfont=dict(color="#DDDDDD"),
        titlefont=dict(color="#DDDDDD"),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#333333",
        zeroline=False, linecolor="#666666", tickfont=dict(color="#DDDDDD"),
        titlefont=dict(color="#DDDDDD"),
    )
    return fig
# =========================================================
# 4. UI
# =========================================================

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
    raw = load_hitters(team)
    df = clean_and_xy(raw)
    batters_all = sorted(df["Batter"].dropna().unique()) if "Batter" in df.columns else []
    if not batters_all:
        st.warning("No batters with batted-ball data for this team.")
        st.stop()

    mode = st.sidebar.radio("Mode", ["Single Batter", "Compare Batters"])

    if mode == "Single Batter":
        batter = st.sidebar.selectbox("Batter", batters_all)
        s_all = df[df["Batter"] == batter].dropna(subset=["X_ft","Y_ft"]).copy()

        tab_spray, tab_heat, tab_bip_zone = st.tabs(["Spray Charts", "Zone Heat Map", "BIP EV/LA Filter"])

        # ---------- Spray tab ----------
        with tab_spray:
            # KPIs
            ev_avg = s_all["ExitSpeed"].mean()
            ev_max = s_all["ExitSpeed"].max()
            la_avg = s_all["Angle"].mean() if ("Angle" in s_all.columns and s_all["Angle"].notna().any()) else np.nan
            hh_rate = (s_all["ExitSpeed"] >= 90).mean() if len(s_all) else np.nan  # 90+ to match score def

            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("BIP", len(s_all))
            c2.metric("Avg EV", f"{ev_avg:.1f} mph" if len(s_all) else "–")
            c3.metric("Max EV", f"{ev_max:.1f} mph" if len(s_all) else "–")
            c4.metric("Avg LA", f"{la_avg:.1f}°" if len(s_all) and pd.notna(la_avg) else "–")
            c5.metric("Hard-Hit %", f"{100*hh_rate:.0f}%" if len(s_all) else "–")

            chart_type = st.radio("Chart Type", ["Field Overlay", "Polar"], horizontal=True)

            if chart_type == "Field Overlay":
                fig = make_field_diamond_figure(s_all, batter_name=batter)
            else:
                req = s_all.dropna(subset=["Direction","Distance"])
                fig = make_polar_spray_figure(req)
            st.plotly_chart(fig, use_container_width=True, theme=None)

            st.subheader("Batted Balls (all BIP)")
            cols = ["Date","Batter","ExitSpeed","Angle","Direction","Distance","PlayResult","X_ft","Y_ft"]
            show = [c for c in cols if c in s_all.columns]
            st.dataframe(s_all[show].sort_values("Date", ascending=False), use_container_width=True)

        # ---------- Zone heatmap tab ----------
        with tab_heat:
            hitter_zones = team_hitter_zone_stats(team)
            if hitter_zones.empty:
                st.info("No zone data available for this team.")
            else:
                try:
                    mat = get_hitter_zone_matrix(hitter_zones, batter)
                    fig_hz = make_zone_heatmap(mat, title=f"{batter} — Hitter Zone Score (0–100)")
                    st.plotly_chart(fig_hz, use_container_width=False)

                    st.caption(
                    "Calculated using xwOBA, xwOBAcon, exit velocity, launch angle, "
                    "Hard-Hit%, Contact%, and Whiff% -- (0 worst - 100 best)"
                    )
                except ValueError as e:
                    st.info(str(e))

        # ---------- NEW: BIP EV/LA filtered zone tab ----------
        with tab_bip_zone:
            plate_df = normalize_plate_coords(raw)
            plate_df = add_swing_flags(plate_df)
            plate_df = add_count_label(plate_df)
            plate_df = standardize_pitch_type(plate_df)

            sub_bip = plate_df[(plate_df["Batter"] == batter) & (plate_df["is_bip"])].copy()
            if sub_bip.empty:
                st.info("No balls in play with plate location data for this hitter.")
            else:
                ev_vals = sub_bip["ExitSpeed"].dropna()
                la_vals = sub_bip["Angle"].dropna()
                if ev_vals.empty or la_vals.empty:
                    st.info("Missing Exit Velocity or Launch Angle data for this hitter.")
                else:
                    ev_min_possible = int(np.floor(ev_vals.min()))
                    ev_max_possible = int(np.ceil(ev_vals.max()))
                    default_ev = 90
                    default_ev = max(ev_min_possible, min(default_ev, ev_max_possible))

                    ev_min = st.slider(
                        "Minimum Exit Velocity (mph)",
                        ev_min_possible,
                        ev_max_possible,
                        value=default_ev,
                    )

                    la_min_possible = int(np.floor(la_vals.min()))
                    la_max_possible = int(np.ceil(la_vals.max()))
                    default_la_low = max(la_min_possible, -10)
                    default_la_high = min(la_max_possible, 40)
                    la_low, la_high = st.slider(
                        "Launch Angle Range (°)",
                        la_min_possible,
                        la_max_possible,
                        value=(default_la_low, default_la_high),
                    )

                    mask = (sub_bip["ExitSpeed"] >= ev_min) & sub_bip["Angle"].between(la_low, la_high)
                    filtered = sub_bip[mask].copy()

                    st.markdown(
                        f"**Filtered BIP:** {len(filtered)} (of {len(sub_bip)} total with plate location)"
                    )

                    if filtered.empty:
                        st.info("No balls in play match the current EV / LA filters.")
                    else:
                        fig_bip = make_hitter_bip_zone_figure(filtered, batter_name=batter)
                        st.plotly_chart(fig_bip, use_container_width=False)
                        st.caption(
                            "Catcher view: negative horizontal = 3B side left, positive = 1B side right."
                        )

    else:
        chosen = st.sidebar.multiselect(
            "Choose up to 12 batters", batters_all,
            default=batters_all[:10], max_selections=12
        )
        if len(chosen) == 0:
            st.info("Select at least one batter.")
            st.stop()

        plot_df = df[df["Batter"].isin(chosen)].dropna(subset=["X_ft","Y_ft"]).copy()
        if plot_df.empty:
            st.info("No XY for selected batters.")
        else:
            hover_cols = [c for c in ["Batter","ExitSpeed","Angle","Direction","Distance","PlayResult","Date"]
                          if c in plot_df.columns]
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
    rawp = load_pitchers(team)
    if rawp.empty or "Pitcher" not in rawp.columns:
        st.warning("No pitcher data for this team.")
        st.stop()

    pitchers = sorted(rawp["Pitcher"].dropna().unique().tolist())
    if not pitchers:
        st.warning("No pitcher data with movement/velo for this team.")
        st.stop()

    pitcher = st.sidebar.selectbox("Pitcher", pitchers)

    # Prepped datasets for different pitcher tabs
    dpp = prep_pitcher_df(rawp)
    plate_p = normalize_plate_coords(rawp)
    plate_p = add_count_label(plate_p)
    plate_p = standardize_pitch_type(plate_p)

    tab_move, tab_heat_p, tab_count = st.tabs(["Movement & Mix", "Zone Heat Map", "Count Zone Map"])

    # ---------- Movement & mix ----------
    with tab_move:
        sub = dpp[dpp["Pitcher"] == pitcher].copy()
        if sub.empty:
            st.info("No movement data for this pitcher.")
        else:
            throws = (sub["PitcherThrows"].dropna().iloc[0] if "PitcherThrows" in sub.columns and sub["PitcherThrows"].notna().any() else "R").upper()
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

    # ---------- Zone heatmap ----------
    with tab_heat_p:
        pitcher_zones = team_pitcher_zone_stats(team)
        if pitcher_zones.empty:
            st.info("No zone data available for this team.")
        else:
            try:
                mat_p = get_pitcher_zone_matrix(pitcher_zones, pitcher)
                fig_pz = make_zone_heatmap(mat_p, title=f"{pitcher} — Pitcher Zone Score (0–100)")
                st.plotly_chart(fig_pz, use_container_width=False)

                st.caption(
                    "Calculated with xwOBA allowed, xwOBAcon allowed, EV allowed, Hard-Hit%(allowed), "
                    "Allowed Launch Angle, CSW%, and Whiff% - (0 worst - 100 best)"
)

            except ValueError as e:
                st.info(str(e))

    # ---------- NEW: Count-by-count zone scatter ----------
    with tab_count:
        subc = plate_p[plate_p["Pitcher"] == pitcher].dropna(
            subset=["PlateLocHeight", "PlateLocSide_norm"]
        ).copy()

        if subc.empty:
            st.info("No pitch location data for this pitcher.")
        else:
            if "CountStr" in subc.columns and subc["CountStr"].notna().any():
                counts_avail = sorted(subc["CountStr"].dropna().unique().tolist())
                options = ["All"] + counts_avail
                default_index = 0
                if "0-0" in counts_avail:
                    default_index = options.index("0-0")
                count_choice = st.selectbox("Count", options, index=default_index)

                if count_choice != "All":
                    subc = subc[subc["CountStr"] == count_choice].copy()
            else:
                count_choice = "All"
                st.caption("No count data available; showing all pitches.")

            st.markdown(f"**Pitches in selection:** {len(subc)}")

            if subc.empty:
                st.info("No pitches match the selected count.")
            else:
                fig_cnt = make_pitcher_count_zone_figure(subc, pitcher_name=pitcher, count_label=count_choice)
                st.plotly_chart(fig_cnt, use_container_width=False)
                st.caption(
                    "Catcher view: negative horizontal = 3B side left, positive = 1B side right. SOLID ball signals a base hit"
                )
