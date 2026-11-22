"""Streamlit Analytics Dashboard: Codeforces Weak-Topic Recommender.

Features:
- User profile & key performance metrics snapshot
- 5-Problem progressive upsolving ladder (TF-IDF + Cosine Similarity)
- Curated Sheets (CP-31 & CSES Problem Set) with persistent progress tracking
- "Where You Got Stuck" Diagnostic Section:
    - Rating Wall detection callout & Attempted vs. Solved grouped bar chart
    - Failure Mode stacked horizontal bar chart (WA, TLE, RE, MLE) with automated diagnoses
    - Abandoned Problems table ("Retry First") with direct solve links
- Visual analytics & Expandable DAG prerequisite topic hierarchy
"""

import math
import re
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analyzer
import cf_api
import recommender

# Set page configuration
st.set_page_config(
    page_title="Codeforces Upsolving Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Sleek & Dark-Mode Friendly CSS
st.markdown(
    """
    <style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }

    /* Ladder & Sheet Cards */
    .ladder-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3);
        transition: all 0.25s ease;
    }
    .ladder-card:hover {
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.15);
        transform: translateY(-2px);
    }
    .ladder-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 10px;
    }
    .step-pill {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .stuck-pill {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.35);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .tier-pill {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .diff-pill {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .diff-easy { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .diff-medium { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .diff-hard { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }

    .problem-title {
        font-size: 1.22rem;
        font-weight: 700;
        color: #f1f5f9;
        text-decoration: none;
    }
    .problem-title:hover {
        color: #38bdf8;
        text-decoration: underline;
    }
    .match-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.35);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* Rating Badges */
    .rating-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 700;
        color: #ffffff;
    }
    .rating-gray { background-color: #6b7280; }
    .rating-green { background-color: #10b981; }
    .rating-cyan { background-color: #06b6d4; }
    .rating-blue { background-color: #3b82f6; }
    .rating-purple { background-color: #a855f7; }
    .rating-orange { background-color: #f97316; }
    .rating-red { background-color: #ef4444; }

    /* Tag Pills */
    .tag-pill {
        display: inline-block;
        background: rgba(51, 65, 85, 0.6);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.2);
        padding: 2px 9px;
        margin: 3px 4px 3px 0;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .primary-tag-pill {
        background: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    /* Action Buttons */
    .solve-btn {
        display: inline-block;
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff !important;
        text-decoration: none !important;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 6px 16px;
        border-radius: 8px;
        border: 1px solid rgba(56, 189, 248, 0.4);
        transition: all 0.2s ease;
    }
    .solve-btn:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
    }
    .retry-btn {
        display: inline-block;
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: #ffffff !important;
        text-decoration: none !important;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 6px 16px;
        border-radius: 8px;
        border: 1px solid rgba(248, 113, 113, 0.4);
        transition: all 0.2s ease;
    }
    .retry-btn:hover {
        background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%);
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
    }

    /* Rating Wall Callout Card */
    .wall-callout {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(153, 27, 27, 0.2) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-left: 6px solid #ef4444;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .wall-callout-clean {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.2) 100%);
        border: 1px solid rgba(16, 185, 129, 0.45);
        border-left: 6px solid #10b981;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }

    /* Status Badges for Topic Ladder */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .status-solved {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.45);
    }
    .status-stuck {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.45);
    }
    .status-todo {
        background: rgba(148, 163, 184, 0.15);
        color: #94a3b8;
        border: 1px solid rgba(148, 163, 184, 0.3);
    }

    /* Google Calendar Button */
    .gcal-btn {
        display: inline-block;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(37, 99, 235, 0.25) 100%);
        color: #93c5fd !important;
        text-decoration: none !important;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 8px;
        border: 1px solid rgba(59, 130, 246, 0.4);
        transition: all 0.2s ease;
    }
    .gcal-btn:hover {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3) 0%, rgba(37, 99, 235, 0.4) 100%);
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
    }

    /* Contest Division Pills */
    .div-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }
    .div-div-4 { background: rgba(20, 184, 166, 0.2); color: #2dd4bf; border: 1px solid rgba(20, 184, 166, 0.45); }
    .div-div-3 { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.45); }
    .div-div-2 { background: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.45); }
    .div-div-1 { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.45); }
    .div-div-1-div-2 { background: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.45); }
    .div-educational { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.45); }
    .div-other { background: rgba(148, 163, 184, 0.2); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.45); }

    /* Best for Your Rating Badge */
    .rec-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.2) 0%, rgba(202, 138, 4, 0.25) 100%);
        color: #fde047;
        border: 1px solid rgba(250, 204, 21, 0.5);
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }

    /* Next Contest Sidebar Banner */
    .next-contest-sidebar {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(30, 41, 59, 0.7) 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-left: 5px solid #38bdf8;
        border-radius: 10px;
        padding: 12px 14px;
        margin-top: 14px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_rating_badge_class(rating: int) -> str:
    """Return CSS class corresponding to Codeforces rating tiers."""
    if rating < 1200:
        return "rating-gray"
    elif rating < 1400:
        return "rating-green"
    elif rating < 1600:
        return "rating-cyan"
    elif rating < 1900:
        return "rating-blue"
    elif rating < 2100:
        return "rating-purple"
    elif rating < 2400:
        return "rating-orange"
    return "rating-red"


def get_division_badge_class(division: str) -> str:
    """Return CSS class corresponding to contest division."""
    div_norm = division.lower().replace(" ", "-").replace(".", "").replace("+", "")
    if "div-4" in div_norm:
        return "div-div-4"
    elif "div-3" in div_norm:
        return "div-div-3"
    elif "div-1-div-2" in div_norm or "div-12" in div_norm or "global" in div_norm:
        return "div-div-1-div-2"
    elif "div-2" in div_norm:
        return "div-div-2"
    elif "div-1" in div_norm:
        return "div-div-1"
    elif "educational" in div_norm:
        return "div-educational"
    return "div-other"


def plot_rating_wall_grouped_chart(
    rating_stats_df: pd.DataFrame,
    rating_wall: Optional[int],
) -> go.Figure:
    """Create a grouped bar chart of Attempted vs. Solved problems across ratings."""
    if rating_stats_df is None or rating_stats_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Rating Stats Data Available", template="plotly_dark")
        return fig

    df = rating_stats_df[rating_stats_df["rating"] >= 800].copy()
    if df.empty:
        df = rating_stats_df.copy()

    x_labels = [str(int(r)) for r in df["rating"]]

    fig = go.Figure()

    # Attempted Problems Bar Trace
    fig.add_trace(
        go.Bar(
            name="Attempted",
            x=x_labels,
            y=df["attempted"],
            marker_color="#64748b",
            hovertemplate="Rating <b>%{x}</b><br>Attempted: <b>%{y}</b><extra></extra>",
        )
    )

    # Solved Problems Bar Trace
    fig.add_trace(
        go.Bar(
            name="Solved",
            x=x_labels,
            y=df["solved"],
            marker_color="#10b981",
            hovertemplate="Rating <b>%{x}</b><br>Solved: <b>%{y}</b><br>Solve Rate: <b>%{customdata:.1%}</b><extra></extra>",
            customdata=df["solve_rate"],
        )
    )

    if rating_wall is not None and str(rating_wall) in x_labels:
        wall_idx = x_labels.index(str(rating_wall))
        fig.add_vline(
            x=wall_idx,
            line_dash="dash",
            line_color="#ef4444",
            line_width=2.5,
            annotation_text=f"🧱 Wall: {rating_wall}",
            annotation_position="top right",
            annotation_font=dict(color="#ef4444", size=11),
        )

    fig.update_layout(
        title=dict(
            text="🧱 <b>Attempted vs. Solved Problems by Rating</b>",
            font=dict(size=15, color="#f8fafc"),
        ),
        barmode="group",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Problem Rating",
            gridcolor="rgba(148, 163, 184, 0.15)",
            color="#94a3b8",
        ),
        yaxis=dict(
            title="Unique Problems",
            gridcolor="rgba(148, 163, 184, 0.15)",
            color="#94a3b8",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1"),
        ),
        margin=dict(l=20, r=20, t=50, b=35),
        height=380,
    )
    return fig


def plot_failure_modes_chart(tag_verdicts_df: pd.DataFrame) -> go.Figure:
    """Create a stacked horizontal bar chart of WA, TLE, RE, and MLE across top 5 weak tags."""
    if tag_verdicts_df is None or tag_verdicts_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Failure Mode Data Available", template="plotly_dark")
        return fig

    df = tag_verdicts_df.iloc[::-1].reset_index(drop=True)
    tags_y = [str(t).title() for t in df["tag"]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Wrong Answer (WA)",
            y=tags_y,
            x=df["WRONG_ANSWER"],
            orientation="h",
            marker_color="#f43f5e",
            hovertemplate="<b>%{y}</b><br>Wrong Answer: <b>%{x}</b><extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Time Limit Exceeded (TLE)",
            y=tags_y,
            x=df["TIME_LIMIT_EXCEEDED"],
            orientation="h",
            marker_color="#f59e0b",
            hovertemplate="<b>%{y}</b><br>Time Limit Exceeded: <b>%{x}</b><extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Runtime Error (RE)",
            y=tags_y,
            x=df["RUNTIME_ERROR"],
            orientation="h",
            marker_color="#8b5cf6",
            hovertemplate="<b>%{y}</b><br>Runtime Error: <b>%{x}</b><extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Memory Limit Exceeded (MLE)",
            y=tags_y,
            x=df["MEMORY_LIMIT_EXCEEDED"],
            orientation="h",
            marker_color="#06b6d4",
            hovertemplate="<b>%{y}</b><br>Memory Limit Exceeded: <b>%{x}</b><extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="⚠️ <b>Failure Modes Across Top 5 Weak Tags</b>",
            font=dict(size=15, color="#f8fafc"),
        ),
        barmode="stack",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Failed Submissions Count",
            gridcolor="rgba(148, 163, 184, 0.15)",
            color="#94a3b8",
        ),
        yaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.05)",
            color="#e2e8f0",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#cbd5e1"),
        ),
        margin=dict(l=10, r=20, t=50, b=35),
        height=380,
    )
    return fig


def plot_weakest_tags_chart(weakness_df: pd.DataFrame) -> go.Figure:
    """Create a horizontal bar chart of the Top 10 weakest tags ranked by weakness_score."""
    if weakness_df is None or weakness_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Weakness Data Available", template="plotly_dark")
        return fig

    top_10 = weakness_df.head(10).copy()
    top_10 = top_10.iloc[::-1].reset_index(drop=True)

    customdata = np.stack(
        (
            top_10["unique_solved"],
            top_10["failed_attempts"],
            top_10["problems_attempted"],
            top_10["solve_rate"],
            top_10["avg_solved_rating"],
        ),
        axis=-1,
    )

    fig = go.Figure(
        go.Bar(
            x=top_10["weakness_score"],
            y=top_10["tag"],
            orientation="h",
            marker=dict(
                color=top_10["weakness_score"],
                colorscale="Reds",
                showscale=False,
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Weakness Score: <b>%{x:.1f}</b><br>"
                "• Unique Solved: <b>%{customdata[0]}</b><br>"
                "• Failed Submissions: <b>%{customdata[1]}</b><br>"
                "• Total Attempted: <b>%{customdata[2]}</b><br>"
                "• Solve Rate: <b>%{customdata[3]:.1%}</b><br>"
                "• Avg Solved Rating: <b>%{customdata[4]:.0f}</b>"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="⚠️ <b>Top 10 Weakest Topics</b> (Ranked by Weakness Score)",
            font=dict(size=15, color="#f8fafc"),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Weakness Score (Higher = Weaker)",
            gridcolor="rgba(148, 163, 184, 0.15)",
            color="#94a3b8",
        ),
        yaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.05)",
            color="#e2e8f0",
        ),
        margin=dict(l=10, r=20, t=45, b=35),
        height=380,
    )
    return fig


def plot_rating_comfort_zone_chart(submissions_df: pd.DataFrame, current_rating: int) -> go.Figure:
    """Create a bar chart of unique solved problems across rating buckets (800 to 2400+)."""
    if submissions_df is None or submissions_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Solved Problems Data", template="plotly_dark")
        return fig

    solved = submissions_df[submissions_df["verdict"] == "OK"].copy()
    solved["rating"] = pd.to_numeric(solved["rating"], errors="coerce")
    solved = solved[solved["rating"].notna() & (solved["rating"] > 0)]
    solved_unique = solved.drop_duplicates(subset=["problem_id"])

    bin_labels = [f"{r}" for r in range(800, 2400, 100)] + ["2400+"]
    bin_counts = {lbl: 0 for lbl in bin_labels}

    for r in solved_unique["rating"]:
        r_val = int(r)
        if r_val < 800:
            bin_counts["800"] += 1
        elif r_val >= 2400:
            bin_counts["2400+"] += 1
        else:
            bucket_val = (r_val // 100) * 100
            bucket_str = str(bucket_val)
            if bucket_str in bin_counts:
                bin_counts[bucket_str] += 1

    x_vals = list(bin_counts.keys())
    y_vals = list(bin_counts.values())

    bar_colors = []
    for x_lbl in x_vals:
        numeric_val = 2400 if x_lbl == "2400+" else int(x_lbl)
        if abs(numeric_val - current_rating) <= 150:
            bar_colors.append("#38bdf8")
        elif numeric_val < current_rating:
            bar_colors.append("#10b981")
        else:
            bar_colors.append("#a855f7")

    fig = go.Figure(
        go.Bar(
            x=x_vals,
            y=y_vals,
            marker=dict(
                color=bar_colors,
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1),
            ),
            hovertemplate="Rating <b>%{x}</b><br>Solved Problems: <b>%{y}</b><extra></extra>",
        )
    )

    snap_current = (
        "2400+"
        if current_rating >= 2400
        else str(max(800, (current_rating // 100) * 100))
    )
    if snap_current in x_vals:
        snap_idx = x_vals.index(snap_current)
        fig.add_vline(
            x=snap_idx,
            line_dash="dash",
            line_color="#f43f5e",
            line_width=2,
            annotation_text=f"Current: {current_rating}",
            annotation_position="top right",
            annotation_font=dict(color="#f43f5e", size=11),
        )

    fig.update_layout(
        title=dict(
            text="📊 <b>Solved Problems by Rating</b> (Rating Comfort Zone)",
            font=dict(size=15, color="#f8fafc"),
        ),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Problem Difficulty Rating",
            gridcolor="rgba(148, 163, 184, 0.15)",
            color="#94a3b8",
        ),
        yaxis=dict(
            title="Unique Problems Solved",
            gridcolor="rgba(148, 163, 184, 0.15)",
            color="#94a3b8",
        ),
        margin=dict(l=20, r=20, t=45, b=35),
        height=380,
    )
    return fig


@st.cache_data(ttl=600, show_spinner=False)
def get_cached_upcoming_contests() -> List[Dict[str, Any]]:
    """Fetch upcoming contests with 10-minute caching."""
    try:
        return cf_api.get_upcoming_contests()
    except Exception:
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_user_info(handle: str) -> Dict[str, Any]:
    """Fetch user profile info with 30-minute caching."""
    return cf_api.get_user_info(handle)


@st.cache_data(ttl=1800, show_spinner=False)
def get_cached_user_submissions(handle: str, count: int = 1000) -> pd.DataFrame:
    """Fetch recent user submissions with 30-minute caching (capped at 1000 for high speed)."""
    return cf_api.get_user_submissions(handle, count=count)


@st.cache_data(ttl=86400, show_spinner=False)
def get_cached_problemset(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch Codeforces problemset with 24-hour memory caching."""
    return cf_api.get_problemset(force_refresh=force_refresh)


def main():
    # Header Banner
    st.title("🎯 Codeforces Weak-Topic Recommender")
    st.markdown(
        "<p style='color: #94a3b8; font-size: 1.05rem; margin-top: -8px;'>"
        "Personalized competitive programming analytics dashboard powered by TF-IDF, "
        "DAG prerequisite boosting, and curated practice sheets.</p>",
        unsafe_allow_html=True,
    )

    # Fetch Upcoming Contests
    upcoming_contests = get_cached_upcoming_contests()

    # 1. Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        handle_input = st.text_input(
            "Codeforces Handle",
            value="tourist",
            placeholder="tourist",
            help="Enter any active Codeforces username",
        ).strip()

        target_stretch = st.slider(
            "Target Rating Stretch",
            min_value=100,
            max_value=400,
            value=300,
            step=50,
            help="Maximum difficulty increase (+100 to +400) for the final ladder step.",
        )

        force_refresh = st.checkbox("Force refresh problemset cache", value=False)
        if force_refresh:
            get_cached_upcoming_contests.clear()
            get_cached_problemset.clear()
            get_cached_user_info.clear()
            get_cached_user_submissions.clear()
            upcoming_contests = get_cached_upcoming_contests()

        generate_btn = st.button("🚀 Generate Upsolving Plan", type="primary", use_container_width=True)

        st.markdown("---")
        st.markdown(
            """
            **Navigation Tabs:**
            - **🪜 Upsolving Ladder**: 5-problem progressive climb
            - **📚 Curated Sheets**: CP-31 Sheet, CSES & CP-31 Topic Ladder
            - **🔍 Stuck Point Diagnostics**: Rating Wall & failure modes
            - **📊 Topic Analytics**: Weak tag ranking & comfort zones
            - **🏆 Upcoming Contests**: Codeforces schedules, divisions & reminders
            """
        )

        # Compact Next Contest Banner in Sidebar
        if upcoming_contests:
            next_c = upcoming_contests[0]
            next_div_cls = get_division_badge_class(next_c["division"])
            st.markdown("---")
            st.markdown(
                f"""
                <div class="next-contest-sidebar">
                    <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: #38bdf8; font-weight: 700; margin-bottom: 4px;">
                        ⚡ Next Contest
                    </div>
                    <div style="font-size: 0.92rem; font-weight: 700; color: #f8fafc; line-height: 1.3; margin-bottom: 6px;">
                        {next_c['name']}
                    </div>
                    <div style="display: flex; gap: 6px; align-items: center; margin-bottom: 6px; flex-wrap: wrap;">
                        <span class="div-pill {next_div_cls}">{next_c['division']}</span>
                        <span style="color: #cbd5e1; font-size: 0.78rem;">⏳ {next_c['countdown']}</span>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.76rem; margin-bottom: 8px;">
                        🗓️ {next_c['start_time_ist']}
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <a href="{next_c['url']}" target="_blank" class="solve-btn" style="padding: 4px 10px; font-size: 0.78rem;">Register 🚀</a>
                        <a href="{next_c['gcal_url']}" target="_blank" class="gcal-btn" style="padding: 4px 10px; font-size: 0.78rem;">📅 Remind</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    active_handle = handle_input if handle_input else "tourist"

    # Fetch Data with Spinner
    with st.spinner(f"Analyzing Codeforces profile for '{active_handle}'..."):
        try:
            user_info = get_cached_user_info(active_handle)
        except cf_api.UserNotFoundError:
            st.error(f"❌ Handle **{active_handle}** was not found on Codeforces. Please check the spelling.")
            return
        except cf_api.NetworkError as e:
            st.error(f"🌐 Network error: {e}. Check internet connection or Codeforces API status.")
            return
        except Exception as e:
            st.error(f"⚠️ Error querying user info: {e}")
            return

        try:
            submissions_df = get_cached_user_submissions(active_handle, count=1000)
        except Exception as e:
            st.error(f"⚠️ Error fetching submissions: {e}")
            return

        try:
            problemset_df = get_cached_problemset(force_refresh=force_refresh)
        except Exception as e:
            st.warning(f"⚠️ Warning: Could not fetch fresh problemset ({e}). Recommendations may be limited.")
            problemset_df = pd.DataFrame()

    # Core Calculations
    user_rating = user_info["rating"]
    weakness_df = analyzer.compute_tag_metrics(submissions_df, user_rating=user_rating)
    priority_tags_df = analyzer.get_priority_weak_tags(weakness_df, top_k=4, return_details=True)
    solved_ids = recommender.get_solved_problem_ids(submissions_df)
    wall_stats = analyzer.get_verdict_and_wall_stats(submissions_df, user_rating=user_rating)
    stuck_problems_df = analyzer.get_stuck_problems(submissions_df)

    # Load Curated Datasets
    cp31_df = recommender.load_cp31_dataset()
    cses_df = recommender.load_cses_dataset()

    # Initialize Session State for CSES Progress
    if "completed_cses_ids" not in st.session_state:
        st.session_state["completed_cses_ids"] = recommender.load_cses_progress()

    # Primary Weakest Tag
    if not priority_tags_df.empty:
        primary_weak_tag = str(priority_tags_df.iloc[0]["tag"]).title()
        primary_weak_score = f"{priority_tags_df.iloc[0]['boosted_score']:.1f}"
    elif not weakness_df.empty:
        primary_weak_tag = str(weakness_df.iloc[0]["tag"]).title()
        primary_weak_score = f"{weakness_df.iloc[0]['weakness_score']:.1f}"
    else:
        primary_weak_tag = "None Detected"
        primary_weak_score = "0.0"

    base_r = max(800, int(round(user_rating / 100.0) * 100))
    target_max_r = base_r + target_stretch

    # 2. Top KPI Row (4 st.metric cards)
    st.markdown("### 📌 Performance Snapshot")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.metric(
            label="Current Rating & Rank",
            value=f"{user_rating}",
            delta=user_info["rank"].title(),
            delta_color="normal",
        )

    with kpi2:
        st.metric(
            label="Total Solved Problems",
            value=f"{len(solved_ids):,}",
            delta=f"{len(submissions_df):,} Submissions",
            delta_color="normal",
        )

    with kpi3:
        st.metric(
            label="Primary Weakest Tag",
            value=primary_weak_tag,
            delta=f"Weakness: {primary_weak_score}",
            delta_color="inverse",
        )

    with kpi4:
        st.metric(
            label="Target Upsolving Range",
            value=f"{base_r} → {target_max_r}",
            delta=f"+{target_stretch} Stretch",
            delta_color="normal",
        )

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

    # 3. Main Navigation Tabs
    tab_ladder, tab_curated, tab_diagnostics, tab_analytics, tab_contests = st.tabs([
        "🪜 5-Problem Upsolving Ladder",
        "📚 Curated Sheets (CP-31 & CSES)",
        "🔍 Where You Got Stuck (Diagnostics)",
        "📊 Detailed Topic Analytics & DAG",
        "🏆 Upcoming Contests",
    ])

    # TAB 1: 5-Problem Upsolving Ladder
    with tab_ladder:
        st.subheader("🪜 Progressive 5-Problem Practice Ladder")
        st.markdown(
            f"<p style='color: #94a3b8; margin-top: -6px;'>"
            f"Ranked via <b>TF-IDF + Cosine Similarity</b> matching your weakest tags with popularity boosting "
            f"across difficulty curve <code>[{base_r}, {base_r + 100}, {base_r + 100}, {base_r + 200}, {target_max_r}]</code>."
            f"</p>",
            unsafe_allow_html=True,
        )

        ladder_problems = recommender.generate_upsolving_ladder(
            problemset_df=problemset_df,
            solved_ids=solved_ids,
            weak_tags_df=priority_tags_df if not priority_tags_df.empty else weakness_df,
            user_rating=user_rating,
            target_stretch=target_stretch,
        )

        if ladder_problems:
            step_descriptions = [
                "Step 1 • Foundation Anchor",
                "Step 2 • Topic Reinforcement",
                "Step 3 • Topic Broadening",
                "Step 4 • Challenge Tier",
                "Step 5 • Mastery Stretch",
            ]

            for idx, prob in enumerate(ladder_problems):
                step_num = prob.get("ladder_step", idx + 1)
                step_desc = (
                    step_descriptions[idx]
                    if idx < len(step_descriptions)
                    else f"Step {step_num}"
                )
                prob_rating = int(prob.get("rating", base_r))
                rating_cls = get_rating_badge_class(prob_rating)
                match_pct = max(1, min(99, int(round(prob.get("combined_score", 0.5) * 100))))
                primary_tag = prob.get("primary_tag", "general")
                prob_url = prob.get("url", f"https://codeforces.com/problemset/problem/{prob.get('problem_id', '')}")

                tag_pills_html = []
                for t in prob.get("tags", []):
                    if t.lower() == primary_tag.lower():
                        tag_pills_html.append(f"<span class='tag-pill primary-tag-pill'>🏷️ {t}</span>")
                    else:
                        tag_pills_html.append(f"<span class='tag-pill'>{t}</span>")
                tags_html_str = "".join(tag_pills_html)

                card_html = f"""
                <div class="ladder-card">
                    <div class="ladder-top-row">
                        <div>
                            <span class="step-pill">{step_desc}</span>
                            <span class="rating-badge {rating_cls}" style="margin-left: 8px;">★ {prob_rating}</span>
                        </div>
                        <div>
                            <span class="match-pill">🎯 {match_pct}% Match</span>
                        </div>
                    </div>
                    <div style="margin: 8px 0 12px 0;">
                        <a href="{prob_url}" target="_blank" class="problem-title">
                            {prob.get('problem_id', '')} — {prob.get('name', 'Unknown Problem')} ↗
                        </a>
                    </div>
                    <div style="margin-bottom: 14px;">
                        {tags_html_str}
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(148, 163, 184, 0.1); padding-top: 10px;">
                        <span style="color: #94a3b8; font-size: 0.82rem;">👥 <b>{prob.get('solvedCount', 0):,}</b> Codeforces Solves</span>
                        <a href="{prob_url}" target="_blank" class="solve-btn">Solve on Codeforces 🚀</a>
                    </div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)
        else:
            st.info("No problems found matching the target rating buckets. Check if the problemset cache is loaded.")

    # TAB 2: Curated Sheets (CP-31 & CSES)
    with tab_curated:
        st.subheader("📚 Curated Practice Ladders: CP-31 Sheet & CSES Problem Set")
        st.markdown(
            "<p style='color: #94a3b8; margin-top: -6px;'>"
            "Complement your custom upsolving plan with globally renowned competitive programming sheets. "
            "CP-31 problems are auto-filtered against your Codeforces history, while CSES progress is persisted locally."
            "</p>",
            unsafe_allow_html=True,
        )

        col_cp31, col_cses = st.columns([1, 1], gap="medium")

        # Column 1: CP-31 Sheet Recommendations
        with col_cp31:
            st.markdown("### 🏆 CP-31 Sheet Recommendations")
            st.markdown(
                f"<p style='color: #94a3b8; font-size: 0.88rem; margin-top: -8px;'>"
                f"Curated for rating range <b>[{user_rating} → {user_rating + 300}]</b>, matching your weak topics:"
                f"</p>",
                unsafe_allow_html=True,
            )

            cp31_recs = recommender.recommend_cp31(
                cp31_df=cp31_df,
                problemset_df=problemset_df,
                solved_ids=solved_ids,
                weak_tags_df=priority_tags_df if not priority_tags_df.empty else weakness_df,
                user_rating=user_rating,
            )

            if not cp31_recs.empty:
                for _, row in cp31_recs.iterrows():
                    p_rating = int(row["rating"])
                    rating_cls = get_rating_badge_class(p_rating)
                    tier_val = int(row.get("sheet_tier", p_rating))

                    p_tags = row["tags"] if isinstance(row["tags"], list) else []
                    matched_tags = row.get("matched_weak_tags", [])
                    matched_set = set(t.lower() for t in matched_tags) if isinstance(matched_tags, list) else set()

                    tag_pills = []
                    for t in p_tags:
                        if t.lower() in matched_set:
                            tag_pills.append(f"<span class='tag-pill primary-tag-pill'>🎯 {t}</span>")
                        else:
                            tag_pills.append(f"<span class='tag-pill'>{t}</span>")
                    tags_html_str = "".join(tag_pills)

                    cp_card = f"""
                    <div class="ladder-card" style="border-left: 5px solid #f59e0b;">
                        <div class="ladder-top-row">
                            <div>
                                <span class="tier-pill">Tier {tier_val}</span>
                                <span class="rating-badge {rating_cls}" style="margin-left: 8px;">★ {p_rating}</span>
                            </div>
                            <div>
                                <span style="color: #94a3b8; font-size: 0.8rem;">👥 {row['solvedCount']:,} solves</span>
                            </div>
                        </div>
                        <div style="margin: 8px 0 10px 0;">
                            <a href="{row['url']}" target="_blank" class="problem-title">
                                {row['problem_id']} — {row['name']} ↗
                            </a>
                        </div>
                        <div style="margin-bottom: 12px;">
                            {tags_html_str}
                        </div>
                        <div style="display: flex; justify-content: flex-end; border-top: 1px solid rgba(148, 163, 184, 0.1); padding-top: 8px;">
                            <a href="{row['url']}" target="_blank" class="solve-btn">Solve on Codeforces 🚀</a>
                        </div>
                    </div>
                    """
                    st.markdown(cp_card, unsafe_allow_html=True)
            else:
                st.info("No unsolved CP-31 problems found in this difficulty bracket.")

        # Column 2: CSES Problem Set Recommendations
        with col_cses:
            st.markdown("### 🌲 CSES Problem Set (Foundational)")
            completed_cses = st.session_state["completed_cses_ids"]
            total_cses_completed = len(completed_cses)

            st.markdown(
                f"<div style='display: flex; justify-content: space-between; align-items: center; margin-top: -8px; margin-bottom: 8px;'>"
                f"<span style='color: #94a3b8; font-size: 0.88rem;'>Mapped from Top 3 weak tags (Easy &rarr; Hard):</span>"
                f"<span style='color: #34d399; font-size: 0.85rem; font-weight: 600;'>✅ {total_cses_completed} Completed</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            cses_recs = recommender.recommend_cses(
                cses_df=cses_df,
                weak_tags_df=priority_tags_df if not priority_tags_df.empty else weakness_df,
                completed_cses_ids=completed_cses,
            )

            if not cses_recs.empty:
                for _, prob in cses_recs.iterrows():
                    prob_id = int(prob["id"])
                    diff_str = str(prob["difficulty"]).lower()
                    diff_cls = f"diff-{diff_str}" if diff_str in ("easy", "medium", "hard") else "diff-medium"

                    cses_card_top = f"""
                    <div class="ladder-card" style="border-left: 5px solid #10b981; margin-bottom: 8px;">
                        <div class="ladder-top-row">
                            <div>
                                <span class="diff-pill {diff_cls}">{prob['difficulty']}</span>
                                <span class="tag-pill" style="margin-left: 8px;">📁 {prob['category']}</span>
                            </div>
                            <div>
                                <span style="color: #64748b; font-size: 0.8rem;">Task #{prob_id}</span>
                            </div>
                        </div>
                        <div style="margin: 6px 0 10px 0;">
                            <a href="{prob['url']}" target="_blank" class="problem-title">
                                {prob['name']} ↗
                            </a>
                        </div>
                    </div>
                    """
                    st.markdown(cses_card_top, unsafe_allow_html=True)

                    # Interactive Mark Done Checkbox
                    col_chk, col_link = st.columns([2, 1])
                    with col_chk:
                        if st.checkbox(f"Mark Task #{prob_id} Completed", key=f"cses_{prob_id}"):
                            st.session_state["completed_cses_ids"].add(prob_id)
                            recommender.save_cses_progress(st.session_state["completed_cses_ids"])
                            st.toast(f"✅ Marked CSES #{prob_id} ({prob['name']}) as Completed!")
                            st.rerun()

                    with col_link:
                        st.markdown(
                            f"<div style='text-align: right;'>"
                            f"<a href='{prob['url']}' target='_blank' style='color: #38bdf8; font-size: 0.85rem; font-weight: 600; text-decoration: none;'>Open CSES ↗</a>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

                # Reset Progress Action
                if total_cses_completed > 0:
                    st.write("")
                    if st.button("🔄 Reset CSES Checkbox Progress", type="secondary"):
                        st.session_state["completed_cses_ids"] = set()
                        recommender.save_cses_progress(set())
                        st.toast("Reset all CSES completed checkboxes.")
                        st.rerun()
            else:
                st.success("🎉 You have completed all foundational CSES tasks in these categories!")

        # Section 3: Topic-Wise CP-31 Rating Ladder
        st.markdown("<div style='margin-top: 28px; margin-bottom: 20px; border-top: 1px solid rgba(148, 163, 184, 0.2);'></div>", unsafe_allow_html=True)
        st.subheader("📈 CP-31 Topic-Wise Rating Ladder")
        st.markdown(
            "<p style='color: #94a3b8; margin-top: -6px;'>"
            "Practice any specific algorithmic tag from the CP-31 sheet in strict ascending order of rating. "
            "Track your completion status, re-attempt problems where you got stuck, and master topics step-by-step."
            "</p>",
            unsafe_allow_html=True,
        )

        # Controls: Topic selectbox, Rating range slider, Hide solved checkbox
        col_topic_sel, col_range_sel, col_hide_sel = st.columns([2.5, 3, 1.5], gap="medium")

        available_tags = recommender.get_cp31_unique_tags(cp31_df, problemset_df)
        if not available_tags:
            available_tags = ["implementation"]

        # Default selectbox to user's #1 weak tag
        default_tag_idx = 0
        if primary_weak_tag:
            target_norm = primary_weak_tag.strip().lower()
            if target_norm in available_tags:
                default_tag_idx = available_tags.index(target_norm)
            else:
                for idx, t in enumerate(available_tags):
                    if target_norm in t or t in target_norm:
                        default_tag_idx = idx
                        break

        with col_topic_sel:
            selected_topic = st.selectbox(
                "Select Topic / Algorithmic Tag",
                options=available_tags,
                index=default_tag_idx,
                format_func=lambda x: f"🏷️ {x.title()}",
                help="Unique algorithmic tags extracted from the merged CP-31 dataset",
            )

        with col_range_sel:
            ladder_min_r, ladder_max_r = st.slider(
                "Rating Range",
                min_value=800,
                max_value=1900,
                value=(800, 1900),
                step=100,
                help="Filter CP-31 problems within this rating bracket (800 to 1900)",
            )

        with col_hide_sel:
            st.write("")
            st.write("")
            hide_solved_problems = st.checkbox("Hide Already Solved", value=False)

        # Fetch Topic Ladder
        ladder_df, ladder_stats = recommender.get_cp31_topic_ladder(
            cp31_df=cp31_df,
            problemset_df=problemset_df,
            submissions_df=submissions_df,
            selected_tag=selected_topic,
            min_rating=ladder_min_r,
            max_rating=ladder_max_r,
        )

        total_ladder = ladder_stats["total_count"]
        solved_ladder = ladder_stats["solved_count"]
        stuck_ladder = ladder_stats["stuck_count"]
        todo_ladder = ladder_stats["todo_count"]
        solve_rate_val = ladder_stats["solve_rate"]

        if total_ladder > 0:
            prog_val = min(1.0, max(0.0, solve_rate_val))
            st.progress(
                prog_val,
                text=f"{selected_topic.title()} Ladder Progress: {solved_ladder} / {total_ladder} Solved ({int(solve_rate_val * 100)}%) | {stuck_ladder} Stuck | {todo_ladder} Todo",
            )
        else:
            st.info(f"No CP-31 problems found for tag '{selected_topic.title()}' between ratings {ladder_min_r} and {ladder_max_r}.")

        # Filter for display if "Hide Already Solved" is active
        display_ladder_df = ladder_df.copy()
        if hide_solved_problems:
            display_ladder_df = display_ladder_df[display_ladder_df["status"] != "Solved"]

        if display_ladder_df.empty and total_ladder > 0 and hide_solved_problems:
            st.success(f"🎉 Incredible work! You have solved all {total_ladder} CP-31 problems for '{selected_topic.title()}' in the {ladder_min_r}–{ladder_max_r} range!")
        elif not display_ladder_df.empty:
            # Group by rating ascending
            unique_ratings = sorted(display_ladder_df["rating"].unique())

            for r_val in unique_ratings:
                r_val_int = int(r_val)
                r_group_df = display_ladder_df[display_ladder_df["rating"] == r_val]
                group_solved = int((r_group_df["status"] == "Solved").sum())
                group_stuck = int((r_group_df["status"] == "Stuck").sum())
                group_todo = int((r_group_df["status"] == "Todo").sum())

                group_count = len(r_group_df)
                plural = "s" if group_count > 1 else ""

                status_summary = []
                if group_solved > 0:
                    status_summary.append(f"{group_solved} Solved")
                if group_stuck > 0:
                    status_summary.append(f"{group_stuck} Stuck")
                if group_todo > 0:
                    status_summary.append(f"{group_todo} Todo")
                summary_str = f" — ({', '.join(status_summary)})" if status_summary else ""

                expander_label = f"★ {r_val_int} Rating: {group_count} Problem{plural}{summary_str}"
                is_expanded = (group_stuck > 0 or group_todo > 0)

                with st.expander(expander_label, expanded=is_expanded):
                    for _, prob_row in r_group_df.iterrows():
                        prob_status = prob_row.get("status", "Todo")
                        prob_rating = int(prob_row.get("rating", r_val_int))
                        prob_tier = int(prob_row.get("sheet_tier", prob_rating))
                        p_cls = get_rating_badge_class(prob_rating)

                        if prob_status == "Solved":
                            status_html = "<span class='status-badge status-solved'>✓ Solved</span>"
                            action_btn = f"<a href='{prob_row['url']}' target='_blank' class='solve-btn' style='background: linear-gradient(135deg, #059669 0%, #047857 100%); border-color: rgba(52, 211, 153, 0.4);'>Review Solution ↗</a>"
                            card_border = "#10b981"
                        elif prob_status == "Stuck":
                            status_html = "<span class='status-badge status-stuck'>⚠️ Stuck</span>"
                            action_btn = f"<a href='{prob_row['url']}' target='_blank' class='retry-btn'>Retry on Codeforces 🚀</a>"
                            card_border = "#ef4444"
                        else:
                            status_html = "<span class='status-badge status-todo'>○ Todo</span>"
                            action_btn = f"<a href='{prob_row['url']}' target='_blank' class='solve-btn'>Solve on Codeforces 🚀</a>"
                            card_border = "#38bdf8"

                        p_tags = prob_row.get("tags", [])
                        if not isinstance(p_tags, list):
                            p_tags = []

                        tag_pills = []
                        for t in p_tags:
                            if t.lower() == selected_topic.lower():
                                tag_pills.append(f"<span class='tag-pill primary-tag-pill'>🎯 {t}</span>")
                            else:
                                tag_pills.append(f"<span class='tag-pill'>{t}</span>")
                        tags_str = "".join(tag_pills)

                        solved_cnt_val = int(prob_row.get("solvedCount", 0))

                        card_html = f"""
                        <div class="ladder-card" style="border-left: 5px solid {card_border}; margin-bottom: 10px;">
                            <div class="ladder-top-row">
                                <div>
                                    {status_html}
                                    <span class="tier-pill" style="margin-left: 8px;">Tier {prob_tier}</span>
                                    <span class="rating-badge {p_cls}" style="margin-left: 8px;">★ {prob_rating}</span>
                                </div>
                                <div>
                                    <span style="color: #94a3b8; font-size: 0.8rem;">👥 {solved_cnt_val:,} solves</span>
                                </div>
                            </div>
                            <div style="margin: 6px 0 10px 0;">
                                <a href="{prob_row['url']}" target="_blank" class="problem-title">
                                    {prob_row['problem_id']} — {prob_row['name']} ↗
                                </a>
                            </div>
                            <div style="margin-bottom: 10px;">
                                {tags_str}
                            </div>
                            <div style="display: flex; justify-content: flex-end; border-top: 1px solid rgba(148, 163, 184, 0.1); padding-top: 8px;">
                                {action_btn}
                            </div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)

    # TAB 3: "Where You Got Stuck" Diagnostic Section
    with tab_diagnostics:
        st.subheader("🔍 Where You Got Stuck: Algorithmic & Rating Bottlenecks")

        # Rating Wall Callout
        wall_rating = wall_stats.get("rating_wall")
        wall_info = wall_stats.get("rating_wall_stats")

        if wall_rating is not None and wall_info:
            solve_pct = int(round(wall_info["solve_rate"] * 100))
            st.markdown(
                f"""
                <div class="wall-callout">
                    <h4 style="margin: 0 0 6px 0; color: #fca5a5;">🧱 Bottleneck Detected: Rating Wall at {wall_rating}</h4>
                    <p style="margin: 0; color: #fecdd3; font-size: 0.96rem;">
                        Your solve rate drops to <b>{solve_pct}%</b> ({wall_info['solved']} solved out of {wall_info['attempted']} attempted problems).
                        This is the lowest difficulty where you encounter steep resistance (&lt; 50% solve rate with &ge; 3 problems attempted).
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="wall-callout-clean">
                    <h4 style="margin: 0 0 6px 0; color: #6ee7b7;">✅ Solid Rating Progression</h4>
                    <p style="margin: 0; color: #a7f3d0; font-size: 0.96rem;">
                        No premature rating wall detected. You maintain at least a 50% solve rate across your attempted difficulty levels.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Two Diagnostic Charts
        col_diag_left, col_diag_right = st.columns(2)

        with col_diag_left:
            fig_wall = plot_rating_wall_grouped_chart(wall_stats["rating_stats_df"], wall_rating)
            st.plotly_chart(fig_wall, use_container_width=True)

        with col_diag_right:
            fig_modes = plot_failure_modes_chart(wall_stats["tag_verdicts_df"])
            st.plotly_chart(fig_modes, use_container_width=True)

        # Automated Diagnoses Callouts
        diagnoses_list = wall_stats.get("automated_diagnoses", [])
        if diagnoses_list:
            st.markdown("#### 💡 Automated Failure Diagnoses:")
            for diag_text in diagnoses_list:
                st.info(f"⚡ {diag_text}")

        st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

        # Abandoned Problems Table ("Retry First")
        st.markdown("### 🔁 Abandoned Problems (Clear Backlog First)")
        st.markdown(
            "<p style='color: #94a3b8; margin-top: -6px;'>"
            "Problems you have attempted multiple times without an accepted verdict. "
            "Re-solving these before climbing the ladder reinforces retention and clears your mental backlog:"
            "</p>",
            unsafe_allow_html=True,
        )

        top_stuck = stuck_problems_df.head(5)
        if not top_stuck.empty:
            for _, row in top_stuck.iterrows():
                p_rating = int(row["rating"]) if pd.notna(row["rating"]) and row["rating"] > 0 else 0
                rating_cls = get_rating_badge_class(p_rating) if p_rating > 0 else "rating-gray"
                rating_label = f"★ {p_rating}" if p_rating > 0 else "Unrated"
                verdict_str = row["most_frequent_verdict"].replace("_", " ")

                p_tags = row["tags"] if isinstance(row["tags"], list) else []
                tag_htmls = "".join([f"<span class='tag-pill'>{t}</span>" for t in p_tags])

                stuck_card = f"""
                <div class="ladder-card" style="border-left: 5px solid #ef4444;">
                    <div class="ladder-top-row">
                        <div>
                            <span class="stuck-pill">⚠️ {row['attempt_count']} Failed Attempts</span>
                            <span class="rating-badge {rating_cls}" style="margin-left: 8px;">{rating_label}</span>
                        </div>
                        <div>
                            <span style="color: #f87171; font-size: 0.82rem; font-weight: 600;">Verdict: {verdict_str}</span>
                        </div>
                    </div>
                    <div style="margin: 8px 0 10px 0;">
                        <a href="{row['url']}" target="_blank" class="problem-title">
                            {row['problem_id']} — {row['name']} ↗
                        </a>
                    </div>
                    <div style="margin-bottom: 12px;">
                        {tag_htmls}
                    </div>
                    <div style="display: flex; justify-content: flex-end; border-top: 1px solid rgba(148, 163, 184, 0.1); padding-top: 8px;">
                        <a href="{row['url']}" target="_blank" class="retry-btn">Retry on Codeforces 🚀</a>
                    </div>
                </div>
                """
                st.markdown(stuck_card, unsafe_allow_html=True)
        else:
            st.success("🎉 **Clean Backlog!** You have zero unsolved attempted problems.")

    # TAB 4: Detailed Topic Analytics & DAG Hierarchy
    with tab_analytics:
        st.subheader("📊 Algorithmic Topic Performance & DAG Hierarchy")

        col_an_left, col_an_right = st.columns(2)
        with col_an_left:
            fig_weak = plot_weakest_tags_chart(weakness_df)
            st.plotly_chart(fig_weak, use_container_width=True)

        with col_an_right:
            fig_comfort = plot_rating_comfort_zone_chart(submissions_df, user_rating)
            st.plotly_chart(fig_comfort, use_container_width=True)

        # Full Topic Weakness Table
        if not weakness_df.empty:
            st.markdown("#### 📋 Comprehensive Topic Metrics:")
            display_weak = weakness_df.copy()
            display_weak["solve_rate"] = (display_weak["solve_rate"] * 100).round(1).astype(str) + "%"
            display_weak = display_weak.rename(
                columns={
                    "tag": "Topic Tag",
                    "problems_attempted": "Attempted",
                    "unique_solved": "Solved",
                    "failed_attempts": "Failed Submissions",
                    "avg_solved_rating": "Avg Solved Rating",
                    "failure_ratio": "Failure Ratio",
                    "solve_rate": "Solve Rate",
                    "rating_deficit": "Rating Deficit",
                    "weakness_score": "Weakness Score",
                }
            )
            st.dataframe(display_weak, use_container_width=True, hide_index=True)

        # Expandable DAG Section
        with st.expander("🕸️ View Prerequisite Topic Graph (DAG) & Boosting Rationale", expanded=True):
            col_dag_left, col_dag_right = st.columns([1, 1])

            with col_dag_left:
                st.markdown("##### 🎯 Active User Prerequisite Boosts:")
                if not priority_tags_df.empty:
                    boosted_found = False
                    for _, row in priority_tags_df.iterrows():
                        reason = row.get("boost_reason", "")
                        if "Prerequisite boost" in reason:
                            boosted_found = True
                            st.success(f"🚀 **{str(row['tag']).upper()}**: {reason}")

                    if not boosted_found:
                        st.info("No multi-tier DAG boosts were triggered. All topics ranked by base weakness.")
                else:
                    st.write("No active topics to evaluate.")

            with col_dag_right:
                st.markdown("##### 📐 Knowledge Graph Hierarchy:")
                st.markdown(
                    """
                    - **Dynamic Programming (`dp`)** $\\rightarrow$ `trees`, `bitmasks`, `graphs`, `probabilities`
                    - **Binary Search (`binary search`)** $\\rightarrow$ `greedy`, `two pointers`, `ternary search`
                    - **Graph Theory (`graphs`)** $\\rightarrow$ `shortest paths`, `dfs and similar`, `dsu`, `flows`
                    - **Mathematics (`math`)** $\\rightarrow$ `number theory`, `combinatorics`, `matrices`, `geometry`
                    - **Data Structures (`data structures`)** $\\rightarrow$ `dsu`, `trees`, `bitmasks`
                    """
                )

    # TAB 5: Upcoming Contests (All Divisions)
    with tab_contests:
        st.subheader("🏆 Upcoming Codeforces Contests (All Divisions)")
        st.markdown(
            "<p style='color: #94a3b8; margin-top: -6px;'>"
            "Never miss a Codeforces round. Live countdowns, IST schedules, "
            "automated rating-tier recommendations, and 1-click Google Calendar reminders."
            "</p>",
            unsafe_allow_html=True,
        )

        if not upcoming_contests:
            st.info("No upcoming scheduled contests found on Codeforces at this moment. Check back soon!")
        else:
            col_f_div, col_f_rec = st.columns([3, 1], gap="medium")

            with col_f_div:
                division_options = ["All", "Div. 4", "Div. 3", "Div. 2", "Div. 1", "Div. 1 + Div. 2", "Educational", "Other"]
                selected_divisions = st.multiselect(
                    "Filter by Division",
                    options=division_options,
                    default=["All"],
                    help="Filter contests by Codeforces round division",
                )

            with col_f_rec:
                st.write("")
                st.write("")
                only_recommended = st.checkbox("⭐ Best for Your Rating Only", value=False)

            # Filter contests
            filtered_contests = []
            for c in upcoming_contests:
                c_div = c["division"]
                is_rec = cf_api.is_contest_recommended_for_rating(c_div, user_rating)
                c_data = dict(c)
                c_data["is_recommended"] = is_rec

                if "All" not in selected_divisions and c_div not in selected_divisions:
                    continue

                if only_recommended and not is_rec:
                    continue

                filtered_contests.append(c_data)

            if not filtered_contests:
                st.info("No contests match your current division filters.")
            else:
                for c in filtered_contests:
                    div_badge_cls = get_division_badge_class(c["division"])
                    is_rec = c["is_recommended"]

                    rec_badge_html = (
                        f"<span class='rec-badge'>⭐ Best for Your Rating ({user_rating})</span>"
                        if is_rec
                        else ""
                    )

                    # Card left border styling
                    if is_rec:
                        card_border = "#eab308"
                    elif c["division"] == "Div. 1":
                        card_border = "#a855f7"
                    elif c["division"] == "Div. 2":
                        card_border = "#6366f1"
                    elif c["division"] == "Div. 3":
                        card_border = "#0284c7"
                    elif c["division"] == "Div. 4":
                        card_border = "#14b8a6"
                    elif c["division"] == "Educational":
                        card_border = "#f59e0b"
                    elif c["division"] == "Div. 1 + Div. 2":
                        card_border = "#f43f5e"
                    else:
                        card_border = "#64748b"

                    card_html = f"""
                    <div class="ladder-card" style="border-left: 5px solid {card_border}; margin-bottom: 14px;">
                        <div class="ladder-top-row">
                            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                                <span class="div-pill {div_badge_cls}">{c['division']}</span>
                                {rec_badge_html}
                                <span style="color: #38bdf8; font-size: 0.85rem; font-weight: 600;">⏳ Starts in {c['countdown']}</span>
                            </div>
                            <div>
                                <span style="color: #94a3b8; font-size: 0.82rem;">⏱️ Duration: <b>{c['duration_formatted']}</b></span>
                            </div>
                        </div>
                        <div style="margin: 8px 0 10px 0;">
                            <a href="{c['url']}" target="_blank" class="problem-title">
                                {c['name']} ↗
                            </a>
                        </div>
                        <div style="display: flex; gap: 16px; align-items: center; margin-bottom: 12px; color: #cbd5e1; font-size: 0.88rem; flex-wrap: wrap;">
                            <span>🗓️ <b>{c['start_time_ist']}</b></span>
                            <span style="color: #64748b;">•</span>
                            <span style="color: #94a3b8;">Contest ID: <b>#{c['id']}</b></span>
                        </div>
                        <div style="display: flex; justify-content: flex-end; gap: 10px; border-top: 1px solid rgba(148, 163, 184, 0.1); padding-top: 10px; flex-wrap: wrap;">
                            <a href="{c['gcal_url']}" target="_blank" class="gcal-btn">📅 Add to Google Calendar</a>
                            <a href="{c['url']}" target="_blank" class="solve-btn">Register on Codeforces 🚀</a>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
