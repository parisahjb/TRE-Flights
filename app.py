import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Disruption Regime Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .metric-card.red   { border-left-color: #d62728; }
    .metric-card.green { border-left-color: #2ca02c; }
    .metric-card.orange{ border-left-color: #ff7f0e; }
    .metric-label { font-size: 0.78rem; color: #6c757d; margin-bottom: 2px; }
    .metric-value { font-size: 1.4rem; font-weight: 700; color: #212529; }
    h1 { color: #212529; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    ews     = pd.read_csv("cell12_early_warning_monitoring.csv")
    ews["Date"] = pd.to_datetime(ews["Date"])
    econ    = pd.read_csv("table2_economic_thresholds.csv")
    sens    = pd.read_csv("table3_sensitivity_analysis.csv")
    grp     = pd.read_csv("cell9_group_summary.csv")
    results = pd.read_csv("results_by_period.csv")
    did     = pd.read_csv("cell13_did_estimates.csv")
    return ews, econ, sens, grp, results, did

ews, econ, sens, grp, results, did = load_data()

TIER_COLORS = {
    "Green":  "#2ca02c",
    "Yellow": "#f0b400",
    "Orange": "#ff7f0e",
    "Red":    "#d62728",
}
REGIME_COLORS = {
    "Financial_Crisis":  "#9467bd",
    "Normal_Operations": "#1f77b4",
    "COVID_Impact":      "#d62728",
    "Recovery":          "#2ca02c",
}
REGIME_LABELS = {
    "Financial_Crisis":  "Financial Crisis (2009–10)",
    "Normal_Operations": "Normal Operations (2011–19)",
    "COVID_Impact":      "COVID Impact (2020–21)",
    "Recovery":          "Recovery (2022–25)",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Airplane_silhouette.svg/240px-Airplane_silhouette.svg.png", width=60)
    st.title("Flight Disruption\nRegime Dashboard")
    st.markdown("---")
    st.markdown("""
    **Study:** 103.7M U.S. domestic flights  
    **Period:** 2009 – 2025  
    **Model:** Random Forest (500 trees)  
    **Training:** 2009–2019 (pre-pandemic)
    """)
    st.markdown("---")
    st.caption(
        "Hajibabaee, P. et al. *Predicting the Unpredictable: "
        "Machine Learning Model Failure During Operational Regime Shifts.* "
        "Transportation Research Part E, 2026."
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📡  Early Warning System",
    "💰  Economic Decision Calculator",
    "🔬  Regime Mechanism Explorer",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — EARLY WARNING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Early Warning System for Regime Shift Detection")
    st.markdown(
        "Three-layer ensemble (CUSUM · Bollinger Bands · Rate-of-Change) monitoring "
        "weekly ROC-AUC across 855 weeks (2009–2025). "
        "The system detected the COVID-19 regime shift **9 weeks** before the WHO declaration."
    )

    # ── Controls ─────────────────────────────────────────────────────────────
    col_ctrl1, col_ctrl2 = st.columns([3, 1])
    with col_ctrl1:
        year_range = st.slider(
            "Year range",
            int(ews["Year"].min()), int(ews["Year"].max()),
            (2017, 2023), step=1,
        )
    with col_ctrl2:
        show_bands = st.checkbox("Show Bollinger Bands", value=True)
        show_annotations = st.checkbox("Show key events", value=True)

    mask = (ews["Year"] >= year_range[0]) & (ews["Year"] <= year_range[1])
    ews_view = ews[mask].copy()

    # ── Summary metrics ───────────────────────────────────────────────────────
    pre  = ews[ews["Year"] <= 2019]
    post = ews[ews["Year"] >= 2021]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card green">
          <div class="metric-label">Green weeks (pre-COVID)</div>
          <div class="metric-value">{(pre["ALERT_TIER"]=="Green").sum()}</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card red">
          <div class="metric-label">Green weeks (2021–2025)</div>
          <div class="metric-value">{(post["ALERT_TIER"]=="Green").sum()}</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        first_alert = ews[(ews["Year"]==2020) & (ews["ALERT_TIER"]!="Green")]["Date"].min()
        st.markdown(f"""
        <div class="metric-card orange">
          <div class="metric-label">First alert in 2020</div>
          <div class="metric-value">{first_alert.strftime("%b %d, %Y")}</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        cusum_min = ews["CUSUM"].min()
        st.markdown(f"""
        <div class="metric-card red">
          <div class="metric-label">CUSUM minimum (2025)</div>
          <div class="metric-value">{cusum_min:.1f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Panel 1: ROC-AUC with Bollinger Bands ────────────────────────────────
    fig_ews = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Weekly ROC-AUC with 20-Week Moving Average",
            "CUSUM Control Chart",
            "Alert Tier Over Time",
        ),
        row_heights=[0.45, 0.30, 0.25],
        vertical_spacing=0.06,
    )

    # AUC raw
    fig_ews.add_trace(go.Scatter(
        x=ews_view["Date"], y=ews_view["AUC"],
        mode="lines", name="Weekly AUC",
        line=dict(color="#aec7e8", width=1),
        opacity=0.7,
    ), row=1, col=1)

    # 20-week MA
    fig_ews.add_trace(go.Scatter(
        x=ews_view["Date"], y=ews_view["AUC_MA20"],
        mode="lines", name="20-week MA",
        line=dict(color="#1f77b4", width=2.5),
    ), row=1, col=1)

    if show_bands:
        fig_ews.add_trace(go.Scatter(
            x=pd.concat([ews_view["Date"], ews_view["Date"][::-1]]),
            y=pd.concat([ews_view["UPPER_BAND"], ews_view["LOWER_BAND"][::-1]]),
            fill="toself",
            fillcolor="rgba(31,119,180,0.10)",
            line=dict(color="rgba(255,255,255,0)"),
            name="±2σ Bollinger Bands",
            showlegend=True,
        ), row=1, col=1)

    # CUSUM
    fig_ews.add_trace(go.Scatter(
        x=ews_view["Date"], y=ews_view["CUSUM"],
        mode="lines", name="CUSUM",
        line=dict(color="#d62728", width=2),
    ), row=2, col=1)
    # Control limits — use value from pre-COVID baseline
    baseline_std = pre["AUC"].std()
    h = 2 * baseline_std
    fig_ews.add_hline(y=0,   line_dash="dash", line_color="gray",  row=2, col=1)
    fig_ews.add_hline(y=-h,  line_dash="dot",  line_color="#d62728",
                      annotation_text=f"Alert threshold (−{h:.3f})",
                      annotation_position="bottom left", row=2, col=1)

    # Alert tiers as coloured dots
    for tier, color in TIER_COLORS.items():
        sub = ews_view[ews_view["ALERT_TIER"] == tier]
        fig_ews.add_trace(go.Scatter(
            x=sub["Date"],
            y=[tier] * len(sub),
            mode="markers",
            marker=dict(color=color, size=6, symbol="circle"),
            name=tier,
            showlegend=(tier in ["Green", "Yellow", "Orange", "Red"]),
        ), row=3, col=1)

    # Key event annotations
    if show_annotations:
        events = [
            ("2020-03-11", "WHO Declaration", "#d62728"),
            ("2022-01-01", "Recovery Start",  "#2ca02c"),
            ("2020-01-06", "First Alert",      "#ff7f0e"),
        ]
        for date_str, label, color in events:
            ev_date = pd.Timestamp(date_str)
            if ews_view["Date"].min() <= ev_date <= ews_view["Date"].max():
                for row_n in [1, 2, 3]:
                    fig_ews.add_vline(
                        x=ev_date, line_dash="dash",
                        line_color=color, opacity=0.6,
                        row=row_n, col=1,
                    )
                fig_ews.add_annotation(
                    x=ev_date, y=1.05,
                    xref="x", yref="paper",
                    text=f"▼ {label}", showarrow=False,
                    font=dict(color=color, size=10),
                )

    fig_ews.update_layout(
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig_ews.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig_ews.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    st.plotly_chart(fig_ews, use_container_width=True)

    # ── Alert tier summary table ──────────────────────────────────────────────
    st.subheader("Alert Tier Summary by Year")
    tier_counts = (
        ews.groupby(["Year", "ALERT_TIER"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Green", "Yellow", "Orange", "Red"], fill_value=0)
        .reset_index()
    )

    def color_tiers(val, col):
        colors_map = {
            "Green": "background-color:#d4edda; color:#155724",
            "Yellow": "background-color:#fff3cd; color:#856404",
            "Orange": "background-color:#ffe5b4; color:#7a4300",
            "Red":    "background-color:#f8d7da; color:#721c24",
        }
        return colors_map.get(col, "")

    styled = tier_counts.style
    for col in ["Green", "Yellow", "Orange", "Red"]:
        if col in tier_counts.columns:
            styled = styled.applymap(
                lambda v, c=col: color_tiers(v, c),
                subset=[col],
            )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    with st.expander("ℹ️ How the alert tiers work"):
        st.markdown("""
        The system combines three binary signals:

        | Layer | Method | Trigger |
        |-------|--------|---------|
        | 1 | CUSUM Control Chart | $S_t^- < -2\\sigma_0$ |
        | 2 | Bollinger Bands (20-week) | AUC falls below lower ±2σ band |
        | 3 | Rate-of-Change | $|\\nabla_t| > 3\\sigma_\\nabla$ |

        **Alert tier** = sum of active signals:
        🟢 Green (0) · 🟡 Yellow (1) · 🟠 Orange (2) · 🔴 Red (3)
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ECONOMIC DECISION CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Economic Decision-Analytic Framework")
    st.markdown(
        "Embed the classifier in a cost matrix and compute the optimal intervention "
        "threshold and expected economic value for your airline's specific cost structure."
    )

    col_inp, col_res = st.columns([1, 2])

    with col_inp:
        st.subheader("Cost Parameters")
        C_d = st.slider(
            "Disruption cost per flight  $C_d$ ($)",
            min_value=2_000, max_value=30_000, value=10_000, step=500,
            format="$%d",
        )
        C_i = st.slider(
            "Intervention cost per flight  $C_i$ ($)",
            min_value=200, max_value=5_000, value=1_500, step=100,
            format="$%d",
        )
        pi = st.slider(
            "Prevention success rate  π",
            min_value=0.10, max_value=0.80, value=0.40, step=0.05,
            format="%.2f",
        )
        st.markdown("---")
        st.markdown(f"""
        **Break-even condition:**  
        Intervention justified when π > C_i / C_d  
        π = **{pi:.2f}** vs. threshold **{C_i/C_d:.3f}**  
        → {"✅ Intervention is economically justified" if pi > C_i/C_d else "❌ Intervention not justified at these costs"}
        """)

    with col_res:
        # ── Compute expected value at various thresholds for each regime ──────
        st.subheader("Expected Value by Regime and Threshold")

        thresholds = np.arange(0.10, 0.90, 0.05)
        regime_order = ["Financial_Crisis", "Normal_Operations",
                        "COVID_Impact", "Recovery"]

        def compute_value(row, tau, Cd, Ci, pi_val):
            total_flights = row["TP"] + row["FP"] + row["TN"] + row["FN"]
            disr_rate     = row["TP"] + row["FN"]   # true positives in full set
            # scale TP/FP/FN by threshold proxy (simple linear interpolation)
            recall_base   = row["Recall"]
            prec_base     = row["Precision"]
            thresh_base   = row["Optimal_Threshold"]
            # Use stored values as single-threshold estimate
            TP = row["TP"]
            FP = row["FP"]
            FN = row["FN"]
            val_tp = TP * (pi_val * Cd - Ci)
            val_fp = -FP * Ci
            val_fn = -FN * Cd
            return (val_tp + val_fp + val_fn) / total_flights

        # Recompute with user params
        ev_rows = []
        for _, row in econ.iterrows():
            v = compute_value(row, row["Optimal_Threshold"], C_d, C_i, pi)
            total = (row["TP"] + row["FP"] + row["TN"] + row["FN"])
            val_total_B = v * total / 1e9
            ev_rows.append({
                "Regime": REGIME_LABELS[row["Period"]],
                "Optimal Threshold": f"{row['Optimal_Threshold']:.2f}",
                "Value / Flight ($)": f"{v:,.0f}",
                "Total Value ($B)":   f"{val_total_B:,.2f}",
                "Recall":             f"{row['Recall']:.3f}",
                "Precision":          f"{row['Precision']:.3f}",
            })

        ev_df = pd.DataFrame(ev_rows)
        st.dataframe(ev_df, use_container_width=True, hide_index=True)

        # ── Sensitivity chart ─────────────────────────────────────────────────
        st.subheader("Sensitivity to Prevention Rate π")

        pi_vals = np.arange(0.10, 0.85, 0.05)
        normal_row = econ[econ["Period"] == "Normal_Operations"].iloc[0]
        total_n = normal_row["TP"] + normal_row["FP"] + normal_row["TN"] + normal_row["FN"]

        values_custom = []
        for p in pi_vals:
            v = compute_value(normal_row, normal_row["Optimal_Threshold"], C_d, C_i, p)
            values_custom.append(v * total_n / 1e9)

        # Also show paper baseline
        values_paper = []
        for p in pi_vals:
            v = compute_value(normal_row, normal_row["Optimal_Threshold"], 10_000, 1_500, p)
            values_paper.append(v * total_n / 1e9)

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=pi_vals, y=values_paper,
            mode="lines+markers",
            name="Paper baseline (C_d=$10K, C_i=$1.5K)",
            line=dict(color="#1f77b4", width=2, dash="dot"),
        ))
        fig_sens.add_trace(go.Scatter(
            x=pi_vals, y=values_custom,
            mode="lines+markers",
            name=f"Your inputs (C_d=${C_d:,}, C_i=${C_i:,})",
            line=dict(color="#d62728", width=2.5),
            marker=dict(size=7),
        ))
        fig_sens.add_vline(x=pi, line_dash="dash", line_color="#ff7f0e",
                           annotation_text=f"Current π={pi:.2f}")
        fig_sens.update_layout(
            xaxis_title="Prevention Success Rate (π)",
            yaxis_title="Total Economic Value ($ Billion, Normal Operations)",
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        fig_sens.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        fig_sens.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_sens, use_container_width=True)

    # ── Confusion matrix breakdown ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Confusion Matrix Breakdown by Regime")

    fig_cm = make_subplots(
        rows=1, cols=4,
        subplot_titles=[REGIME_LABELS[p] for p in regime_order],
    )
    for i, period in enumerate(regime_order):
        row_data = econ[econ["Period"] == period].iloc[0]
        cm_vals = np.array([
            [row_data["TN"], row_data["FP"]],
            [row_data["FN"], row_data["TP"]],
        ])
        fig_cm.add_trace(go.Heatmap(
            z=cm_vals / cm_vals.sum() * 100,
            x=["Predicted Normal", "Predicted Disrupted"],
            y=["Actual Normal", "Actual Disrupted"],
            colorscale="Blues",
            showscale=(i == 3),
            text=[[f"{v/1e6:.1f}M<br>({v/cm_vals.sum()*100:.1f}%)"
                   for v in row] for row in cm_vals],
            texttemplate="%{text}",
            textfont=dict(size=9),
            zmin=0, zmax=100,
        ), row=1, col=i+1)

    fig_cm.update_layout(height=320, margin=dict(t=60, b=20))
    st.plotly_chart(fig_cm, use_container_width=True)

    with st.expander("ℹ️ Cost model formula"):
        st.latex(r"""
        C(\tau) = TP \cdot (C_i - \pi C_d) + FP \cdot C_i + FN \cdot C_d
        """)
        st.markdown("""
        - **TP** — True positives: intervene on actual disruption → net saving if π·C_d > C_i  
        - **FP** — False positives: unnecessary intervention costs C_i  
        - **FN** — False negatives: missed disruptions cost full C_d
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REGIME MECHANISM EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Disruption Mechanism Changes Across Regimes")
    st.markdown(
        "Feature importance shifts reveal *how* disruption mechanisms changed across "
        "economic regimes — even when overall AUC remained stable."
    )

    col_l, col_r = st.columns(2)

    # ── Feature group trajectories ────────────────────────────────────────────
    with col_l:
        st.subheader("Feature Group Importance Trajectories")
        regimes = ["Financial_Crisis", "Normal_Operations", "COVID_Impact", "Recovery"]
        regime_short = ["Financial\nCrisis", "Normal\nOperations", "COVID\nImpact", "Recovery"]

        groups_to_show = st.multiselect(
            "Select feature groups",
            options=grp["Group"].tolist(),
            default=["Cascading", "Historical", "Temporal"],
        )

        fig_traj = go.Figure()
        group_styles = {
            "Cascading":  dict(color="#d62728", symbol="circle",   width=2.5),
            "Historical": dict(color="#1f77b4", symbol="square",   width=2.5),
            "Temporal":   dict(color="#2ca02c", symbol="triangle-up", width=2.5),
            "Spatial":    dict(color="#9467bd", symbol="diamond",  width=2),
            "Carrier":    dict(color="#8c564b", symbol="cross",    width=2),
            "Other":      dict(color="#7f7f7f", symbol="x",        width=1.5),
        }

        for grp_name in groups_to_show:
            row_g = grp[grp["Group"] == grp_name].iloc[0]
            vals  = [row_g[r] for r in regimes]
            style = group_styles.get(grp_name, dict(color="#333", symbol="circle", width=2))
            fig_traj.add_trace(go.Scatter(
                x=regime_short, y=vals,
                mode="lines+markers",
                name=grp_name,
                line=dict(color=style["color"], width=style["width"]),
                marker=dict(symbol=style["symbol"], size=10, color=style["color"]),
            ))

        # COVID shading
        fig_traj.add_vrect(
            x0="Normal\nOperations", x1="COVID\nImpact",
            fillcolor="rgba(214,39,40,0.08)",
            layer="below", line_width=0,
            annotation_text="COVID-19", annotation_position="top left",
            annotation_font_color="#d62728",
        )
        fig_traj.update_layout(
            xaxis_title="Economic Regime",
            yaxis_title="Feature Group Importance (MDI)",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        fig_traj.update_xaxes(showgrid=False)
        fig_traj.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_traj, use_container_width=True)

    # ── % change from Normal Operations ───────────────────────────────────────
    with col_r:
        st.subheader("% Change from Normal Operations Baseline")

        compare_regime = st.selectbox(
            "Compare against Normal Operations:",
            ["COVID_Impact", "Financial_Crisis", "Recovery"],
            index=0,
        )

        normal_vals  = grp.set_index("Group")["Normal_Operations"]
        compare_vals = grp.set_index("Group")[compare_regime]
        pct_change   = ((compare_vals - normal_vals) / normal_vals * 100).sort_values()

        colors_bar = ["#d62728" if v < 0 else "#1f77b4" for v in pct_change]

        fig_bar = go.Figure(go.Bar(
            x=pct_change.values,
            y=pct_change.index,
            orientation="h",
            marker_color=colors_bar,
            text=[f"{v:+.1f}%" for v in pct_change.values],
            textposition="outside",
        ))
        fig_bar.add_vline(x=0, line_color="black", line_width=1.5)
        fig_bar.update_layout(
            xaxis_title="% Change from Normal Operations",
            height=400,
            margin=dict(l=80, r=80),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        fig_bar.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Model performance table ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Model Performance Across Regimes")

    col_perf1, col_perf2 = st.columns([2, 1])

    with col_perf1:
        fig_auc = go.Figure()
        for _, row in results.iterrows():
            color = REGIME_COLORS.get(row["Period"], "#333")
            label = REGIME_LABELS.get(row["Period"], row["Period"])
            border = "dash" if row["Sample_Type"] == "Out-of-sample" else "solid"
            fig_auc.add_trace(go.Bar(
                x=[label],
                y=[row["ROC_AUC"]],
                name=label,
                marker_color=color,
                marker_line=dict(width=2, color="white"),
                text=f'{row["ROC_AUC"]:.3f}',
                textposition="outside",
                width=0.5,
            ))
        fig_auc.add_hline(y=0.5, line_dash="dot", line_color="gray",
                          annotation_text="Random baseline (0.5)")
        fig_auc.update_layout(
            title="ROC-AUC by Regime (dashed border = out-of-sample)",
            yaxis=dict(range=[0.4, 0.80], title="ROC-AUC"),
            showlegend=False,
            height=320,
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_auc, use_container_width=True)

    with col_perf2:
        perf_table = results[[
            "Period", "Sample_Type", "ROC_AUC", "F1_Score", "Disruption_Rate"
        ]].copy()
        perf_table["Period"] = perf_table["Period"].map(REGIME_LABELS)
        perf_table.columns = ["Regime", "Sample", "AUC", "F1", "Disr. Rate"]
        perf_table["AUC"]   = perf_table["AUC"].apply(lambda x: f"{x:.3f}")
        perf_table["F1"]    = perf_table["F1"].apply(lambda x: f"{x:.3f}")
        perf_table["Disr. Rate"] = perf_table["Disr. Rate"].apply(lambda x: f"{x:.1%}")
        st.dataframe(perf_table, use_container_width=True, hide_index=True)

    # ── DiD results ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Causal Evidence: Difference-in-Differences (Hub vs. Point-to-Point)")

    col_did1, col_did2 = st.columns(2)

    with col_did1:
        fig_did = go.Figure()
        for _, row in did.iterrows():
            color = "#d62728" if row["Airline_Type"] == "Hub" else "#1f77b4"
            label = ("Hub-and-Spoke (Treated)"
                     if row["Airline_Type"] == "Hub"
                     else "Point-to-Point (Control)")
            fig_did.add_trace(go.Scatter(
                x=["Normal Operations", "COVID Impact"],
                y=[row["Normal_Cascading"], row["COVID_Cascading"]],
                mode="lines+markers+text",
                name=label,
                line=dict(color=color, width=3),
                marker=dict(size=12, color=color),
                text=[f"{row['Normal_Cascading']:.3f}",
                      f"{row['COVID_Cascading']:.3f}"],
                textposition=["top center", "bottom center"],
            ))
        fig_did.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text=f"DiD = −12.8%<br>(95% CI: [−18.4%, −7.2%], p<0.01)",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d62728",
            borderwidth=1.5,
            font=dict(size=11, color="#d62728"),
        )
        fig_did.update_layout(
            title="Cascading Feature Importance: Hub vs. P2P",
            yaxis_title="Cascading Feature Importance",
            height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_did, use_container_width=True)

    with col_did2:
        st.markdown("#### DiD Results")
        for _, row in did.iterrows():
            label = ("Hub-and-Spoke" if row["Airline_Type"] == "Hub"
                     else "Point-to-Point")
            color_class = "red" if row["Airline_Type"] == "Hub" else ""
            st.markdown(f"""
            <div class="metric-card {color_class}">
              <div class="metric-label">{label} — Percent Change (Normal → COVID)</div>
              <div class="metric-value">{row['Percent_Change']:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        hub_pct = did[did["Airline_Type"]=="Hub"]["Percent_Change"].values[0]
        p2p_pct = did[did["Airline_Type"]=="P2P"]["Percent_Change"].values[0]
        did_est = hub_pct - p2p_pct

        st.markdown(f"""
        <div class="metric-card red">
          <div class="metric-label">DiD Estimate β̂_DiD</div>
          <div class="metric-value">{did_est:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        **Interpretation:** Hub airlines experienced a significantly 
        larger decline in network-driven disruption mechanisms than 
        point-to-point carriers, providing causal evidence that 
        network topology mediates regime-dependent vulnerability.
        
        Bootstrap CI: [−18.4%, −7.2%], *p* < 0.01, *R* = 1,000 replications.
        """)
